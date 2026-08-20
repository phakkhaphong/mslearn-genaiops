"""
Complete Cloud Evaluation Script for Trail Guide Agent

Runs the full evaluation pipeline:
1. Uploads evaluation dataset to Microsoft Foundry
2. Creates evaluation definition with quality evaluators
3. Runs evaluation and polls for completion
4. Retrieves and displays results

Evaluates: Intent Resolution, Relevance, and Groundedness
"""

import argparse
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

from dotenv import load_dotenv
from azure.identity import DefaultAzureCredential
from azure.ai.projects import AIProjectClient
from openai.types.eval_create_params import DataSourceConfigCustom
from openai.types.evals.create_eval_jsonl_run_data_source_param import (
    CreateEvalJSONLRunDataSourceParam,
    SourceFileID,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

load_dotenv()  # reads variables from the .env file in your project root

endpoint              = os.environ.get("AZURE_AI_PROJECT_ENDPOINT")
model_deployment_name = os.environ.get("MODEL_NAME", "gpt-5.1")
dataset_name          = "trail-guide-evaluation-dataset"
dataset_version       = "1"

# The script writes a plain-text summary here when it finishes.
# This file is committed to the branch so the GitHub Actions workflow
# can read it and post results as a PR comment — no re-running needed.
RESULTS_FILE = Path("evaluation_results.txt")

# How long to keep retrying 404s right after create (Foundry can take a
# few seconds before GET /runs/{id} becomes consistent).
NOT_FOUND_RETRY_SECONDS = 180
POLL_INTERVAL_SECONDS = 10

# Lazily initialized in init_clients() so unit tests can import this module
# without talking to Azure.
project_client = None
client = None


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def section(title: str) -> None:
    """Print a clearly visible section header to make the log easy to skim."""
    print(f"\n{'=' * 80}")
    print(f"{title}")
    print(f"{'=' * 80}")


def init_clients() -> None:
    """Create Foundry / OpenAI clients. Exits if the project endpoint is missing."""
    global project_client, client
    if client is not None:
        return

    if not endpoint:
        print("ERROR: AZURE_AI_PROJECT_ENDPOINT is not set.")
        print("       Add it to your .env file and try again.")
        sys.exit(1)

    project_client = AIProjectClient(
        endpoint=endpoint,
        credential=DefaultAzureCredential(),
    )
    client = project_client.get_openai_client()


def parse_args(argv=None) -> argparse.Namespace:
    """Parse CLI flags used to resume an in-flight cloud evaluation run.

    Args:
        argv: Optional argument list. Defaults to ``sys.argv[1:]``.

    Returns:
        Parsed arguments. ``eval_id`` / ``run_id`` may be None when starting
        a new evaluation.
    """
    parser = argparse.ArgumentParser(
        description="Run or resume Trail Guide Agent cloud evaluation.",
    )
    parser.add_argument(
        "--eval-id",
        default=os.environ.get("EVAL_ID"),
        help="Existing evaluation definition ID (skip create; resume polling).",
    )
    parser.add_argument(
        "--run-id",
        default=os.environ.get("EVAL_RUN_ID"),
        help="Existing evaluation run ID (skip create; resume polling).",
    )
    return parser.parse_args(argv)


def is_not_found(exc: Exception) -> bool:
    """Return True when the exception is an HTTP 404 / resource-not-found error.

    Args:
        exc: Exception raised by the OpenAI-compatible client.

    Returns:
        True if the error should be retried as a transient visibility lag.
    """
    if getattr(exc, "status_code", None) == 404:
        return True
    message = str(exc).lower()
    return "not found" in message or "notfound" in message


def resume_command(eval_id: str, run_id: str) -> str:
    """Return the command used to resume polling a live cloud run.

    Args:
        eval_id: Evaluation definition ID.
        run_id: Evaluation run ID.

    Returns:
        Shell command string.
    """
    return (
        "python src/evaluators/evaluate_agent.py "
        f"--eval-id {eval_id} --run-id {run_id}"
    )


def write_in_progress(eval_id: str, run_id: str) -> None:
    """Persist resume IDs so a local crash does not look like a finished eval.

    Args:
        eval_id: Evaluation definition ID.
        run_id: Evaluation run ID.
    """
    RESULTS_FILE.write_text(
        "\n".join(
            [
                "=" * 80,
                " Trail Guide Agent - Evaluation IN PROGRESS",
                "=" * 80,
                "",
                f"  Eval ID : {eval_id}",
                f"  Run ID  : {run_id}",
                "",
                "Cloud scoring is still running. This file is not the final scorecard.",
                "Resume polling (do not start a new evaluation):",
                f"  {resume_command(eval_id, run_id)}",
                "",
            ]
        ),
        encoding="utf-8",
    )


def fetch_eval_run(openai_client, eval_id: str, run_id: str):
    """Fetch a run, falling back to list() when retrieve returns 404.

    Args:
        openai_client: OpenAI-compatible Foundry client.
        eval_id: Evaluation definition ID.
        run_id: Evaluation run ID.

    Returns:
        The matching run object.

    Raises:
        Exception: Re-raised when retrieve fails for a non-404 reason, or when
            both retrieve and list cannot see the run yet.
    """
    try:
        return openai_client.evals.runs.retrieve(run_id=run_id, eval_id=eval_id)
    except Exception as exc:
        if not is_not_found(exc):
            raise
        for run in openai_client.evals.runs.list(eval_id=eval_id):
            if getattr(run, "id", None) == run_id:
                return run
        raise


# ---------------------------------------------------------------------------
# Step 1 – Upload the evaluation dataset
# ---------------------------------------------------------------------------

def upload_dataset() -> str:
    """
    Upload the JSONL evaluation dataset to Azure AI Foundry and return its ID.

    The dataset is a list of JSON objects, each containing:
      - query        : the user question sent to the agent
      - response     : the agent's answer
      - ground_truth : the expected correct answer (used by some evaluators)
    """
    section("Step 1: Uploading evaluation dataset")

    dataset_path = (
        Path(__file__).parent.parent.parent
        / "data"
        / "trail_guide_evaluation_dataset.jsonl"
    )

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {dataset_path}.\n"
            "Make sure you are running the script from the repository root."
        )

    print(f"\nDataset: {dataset_path.name}")
    print("Uploading...")

    try:
        data_id = project_client.datasets.upload_file(
            name=dataset_name,
            version=dataset_version,
            file_path=str(dataset_path),
        ).id
        print(f"\n✓ Dataset uploaded successfully")

    except Exception as upload_error:
        # If this version was already uploaded in a previous run, reuse it.
        # Foundry does not allow uploading the same name+version twice.
        if "already exists" in str(upload_error):
            print(f"\n  Dataset version {dataset_version} already exists in Foundry.")
            print(f"  Retrieving existing dataset ID...")
            dataset_obj = project_client.datasets.get(name=dataset_name, version=dataset_version)
            data_id = dataset_obj.id
            print(f"  ✓ Using existing dataset")
        else:
            # Any other upload error is unexpected — re-raise so main() catches it
            raise

    print(f"  Dataset ID: {data_id}")
    return data_id


# ---------------------------------------------------------------------------
# Step 2 – Create the evaluation definition
# ---------------------------------------------------------------------------

def create_evaluation_definition():
    """
    Register an evaluation definition in Foundry.

    This tells the platform:
      - What data schema to expect (query / response / ground_truth)
      - Which built-in evaluators to run and how to map dataset fields to them

    Returns the evaluation object (its ID is needed in later steps).
    """
    section("Step 2: Creating evaluation definition")

    print(f"\nConfiguration:")
    print(f"  Judge Model: {model_deployment_name}")
    print(f"  Evaluators: Intent Resolution, Relevance, Groundedness")

    # Tell Foundry the shape of each record in the dataset
    data_source_config = DataSourceConfigCustom(
        type="custom",
        item_schema={
            "type": "object",
            "properties": {
                "query":        {"type": "string"},
                "response":     {"type": "string"},
                "ground_truth": {"type": "string"},
            },
            "required": ["query", "response", "ground_truth"],
        },
    )

    # Each entry names a built-in evaluator and maps dataset columns to its
    # expected parameters using the {{item.<column>}} template syntax.
    testing_criteria = [
        {
            "type": "azure_ai_evaluator",
            "name": "intent_resolution",
            "evaluator_name": "builtin.intent_resolution",
            "initialization_parameters": {"deployment_name": model_deployment_name},
            "data_mapping": {
                "query":    "{{item.query}}",
                "response": "{{item.response}}",
            },
        },
        {
            "type": "azure_ai_evaluator",
            "name": "relevance",
            "evaluator_name": "builtin.relevance",
            "initialization_parameters": {"deployment_name": model_deployment_name},
            "data_mapping": {
                "query":    "{{item.query}}",
                "response": "{{item.response}}",
            },
        },
        {
            "type": "azure_ai_evaluator",
            "name": "groundedness",
            "evaluator_name": "builtin.groundedness",
            "initialization_parameters": {"deployment_name": model_deployment_name},
            "data_mapping": {
                "query":    "{{item.query}}",
                "response": "{{item.response}}",
                "context":  "{{item.ground_truth}}",
            },
        },
    ]

    print("\nCreating evaluation...")
    eval_object = client.evals.create(
        name="Trail Guide Quality Evaluation",
        data_source_config=data_source_config,
        testing_criteria=testing_criteria,
    )

    print(f"\n✓ Evaluation definition created")
    print(f"  Evaluation ID: {eval_object.id}")
    return eval_object


# ---------------------------------------------------------------------------
# Step 3 – Start the evaluation run
# ---------------------------------------------------------------------------

def run_evaluation(eval_object, data_id):
    """
    Start a cloud evaluation run that scores every record in the dataset.

    The run is asynchronous — Foundry processes items in parallel in the cloud.
    We get a run object immediately; its status starts as 'queued' or 'running'.
    """
    section("Step 3: Running cloud evaluation")

    eval_run = client.evals.runs.create(
        eval_id=eval_object.id,
        name="trail-guide-baseline-eval",
        data_source=CreateEvalJSONLRunDataSourceParam(
            type="jsonl",
            source=SourceFileID(
                type="file_id",
                id=data_id,
            ),
        ),
    )

    print(f"\n✓ Evaluation run started")
    print(f"  Run ID: {eval_run.id}")
    print(f"  Status: {eval_run.status}")
    print(f"\nThis may take 15-60+ minutes for 89 items depending on capacity and quota...")
    return eval_run


# ---------------------------------------------------------------------------
# Step 4 – Poll until the run finishes
# ---------------------------------------------------------------------------

def poll_for_results(
    eval_object,
    eval_run,
    openai_client=None,
    sleep_fn=time.sleep,
    poll_interval=POLL_INTERVAL_SECONDS,
    not_found_retry_seconds=NOT_FOUND_RETRY_SECONDS,
):
    """Poll until the cloud run reaches a terminal status.

    Foundry sometimes returns 404 ``Project not found`` on the first GET after
    create even though the run is already queued. Retry those 404s and fall
    back to ``runs.list`` instead of treating the local poll error as a failed
    evaluation.

    Args:
        eval_object: Evaluation definition (needs ``id``).
        eval_run: Evaluation run (needs ``id``).
        openai_client: OpenAI-compatible client. Defaults to the module client.
        sleep_fn: Sleep callable; injectable for tests.
        poll_interval: Seconds between polls.
        not_found_retry_seconds: How long to retry 404s before giving up.

    Returns:
        The final run object (contains report URL and aggregate counts).

    Raises:
        RuntimeError: If the cloud run fails/cancels, or 404s persist too long.
    """
    section("Step 4: Polling for completion")
    openai_client = openai_client or client
    start_time = time.time()

    while True:
        elapsed = int(time.time() - start_time)
        try:
            run = fetch_eval_run(openai_client, eval_object.id, eval_run.id)
        except Exception as exc:
            if is_not_found(exc) and elapsed < not_found_retry_seconds:
                print(
                    f"  [{elapsed}s] Status: waiting (run not visible yet, retrying)",
                    end="\r",
                    flush=True,
                )
                sleep_fn(poll_interval)
                continue
            raise RuntimeError(
                f"Could not poll evaluation run after {elapsed}s.\n"
                f"  Eval ID : {eval_object.id}\n"
                f"  Run ID  : {eval_run.id}\n"
                f"  Error   : {exc}\n"
                f"  The cloud run may still be scoring. Resume with:\n"
                f"  {resume_command(eval_object.id, eval_run.id)}"
            ) from exc

        status = getattr(run, "status", None)
        if status == "completed":
            print(f"\n\n✓ Evaluation completed in {elapsed} seconds")
            return run
        if status == "failed":
            error_detail = getattr(run, "error", None) or "No additional details available."
            raise RuntimeError(
                f"Evaluation run failed after {elapsed}s.\n"
                f"  Eval ID : {eval_object.id}\n"
                f"  Run ID  : {eval_run.id}\n"
                f"  Error   : {error_detail}\n"
                f"  To inspect: open Azure AI Foundry portal > Evaluations"
            )
        if status in {"canceled", "cancelled"}:
            raise RuntimeError(
                f"Evaluation run was canceled after {elapsed}s.\n"
                f"  Eval ID : {eval_object.id}\n"
                f"  Run ID  : {eval_run.id}"
            )

        print(f"  [{elapsed}s] Status: {status}", end="\r", flush=True)
        sleep_fn(poll_interval)


# ---------------------------------------------------------------------------
# Step 5 – Collect scores and save results
# ---------------------------------------------------------------------------

def retrieve_and_display_results(eval_object, run):
    """
    Fetch per-item evaluator outputs, compute aggregate statistics, print a
    human-readable summary, and write the same summary to RESULTS_FILE.

    Scores are on a 1-5 scale; a score >= 3 is considered a pass.

    The written file is intended to be committed to the branch so the
    GitHub Actions workflow can read it without re-running the evaluation.

    Returns the raw list of output items for any further inspection.
    """
    section("Step 5: Retrieving results")

    print(f"\nEvaluation Summary")

    # Retrieve every scored item from the run
    output_items = list(
        client.evals.runs.output_items.list(
            run_id=run.id,
            eval_id=eval_object.id,
        )
    )

    # Separate items by status so we can report errors alongside scores
    errored_items = [
        item for item in output_items
        if getattr(item, "status", None) == "error"
    ]
    scored_items = [
        item for item in output_items
        if getattr(item, "status", None) != "error"
    ]

    if errored_items:
        print(f"\n  ⚠ {len(errored_items)} item(s) errored during evaluation.")
        print(f"    First error: {getattr(errored_items[0], 'error', 'details unavailable')}")
        print(f"    Open Azure AI Foundry portal > Evaluations to inspect all failed items.")

    # Collect individual scores grouped by metric name
    scores: dict[str, list[float]] = {
        "intent_resolution": [],
        "relevance":         [],
        "groundedness":      [],
    }

    for item in scored_items:
        if hasattr(item, "evaluator_outputs"):
            for output in item.evaluator_outputs:
                if output.name in scores and hasattr(output, "score"):
                    scores[output.name].append(output.score)

    # --- Build summary text (printed to console and written to file) ---
    # Everything written to `lines` ends up both on screen and in the file,
    # so the file always has useful content regardless of whether scores loaded.

    metric_labels = {
        "intent_resolution": "Intent Resolution",
        "relevance":         "Relevance        ",
        "groundedness":      "Groundedness     ",
    }

    lines = [
        "=" * 80,
        " Trail Guide Agent - Evaluation Results",
        "=" * 80,
        f"\n  Eval ID      : {eval_object.id}",
        f"  Run ID       : {run.id}",
        f"  Total items  : {len(output_items)}",
        f"  Errored items: {len(errored_items)}",
        f"  Scored items : {len(scored_items)}",
        "\nAverage Scores (1-5 scale, threshold: 3)",
    ]

    any_scores = False
    pass_lines = ["\nPass Rates (score >= 3)"]

    for key, label in metric_labels.items():
        values = scores[key]
        if values:
            any_scores = True
            avg  = sum(values) / len(values)
            rate = sum(1 for v in values if v >= 3) / len(values) * 100
            lines.append(f"  {label}: {avg:.2f} (n={len(values)})")
            pass_lines.append(f"  {label}: {rate:.1f}%")

    if not any_scores:
        # Scores missing — the evaluation may have completed but returned no
        # evaluator_outputs. Open the Report URL above to inspect in the portal.
        lines.append("  No scores returned — open Azure AI Foundry portal > Evaluations for details.")
        pass_lines.append("  No scores returned.")

    lines.extend(pass_lines)
    summary = "\n".join(lines)

    print(summary)

    # Write to file so the GitHub Actions 'comment' job can read it directly
    RESULTS_FILE.write_text(summary, encoding="utf-8")
    print(f"\n  Results saved to {RESULTS_FILE}")
    print(f"  Commit this file so the GitHub Actions workflow can read it.")

    # Emit report_url as a GitHub Actions step output when running in CI
    report_url = getattr(run, "report_url", None) or (
        f"{endpoint.rstrip('/')}/evaluations/{eval_object.id}/runs/{run.id}"
    )
    github_output = os.environ.get("GITHUB_OUTPUT")
    if github_output:
        with open(github_output, "a", encoding="utf-8") as gh_out:
            gh_out.write(f"report_url={report_url}\n")
        print(f"  GitHub Actions output set: report_url={report_url}")

    return output_items


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None) -> None:
    """Orchestrate the full evaluation pipeline, or resume an existing run.

    Args:
        argv: Optional CLI arguments. Defaults to ``sys.argv[1:]``.
    """
    args = parse_args(argv)
    resume = bool(args.eval_id or args.run_id)
    if resume and not (args.eval_id and args.run_id):
        print("ERROR: --eval-id and --run-id must be provided together to resume.")
        sys.exit(2)

    init_clients()

    section(" Trail Guide Agent - Cloud Evaluation")
    print(f"\nConfiguration:")
    print(f"  Project: {endpoint}")
    print(f"  Model:   {model_deployment_name}")
    print(f"  Dataset: {dataset_name} (v{dataset_version})")
    if resume:
        print(f"  Mode:    resume")
        print(f"  Eval ID: {args.eval_id}")
        print(f"  Run ID:  {args.run_id}")

    eval_object = None
    eval_run = None
    try:
        if resume:
            eval_object = SimpleNamespace(id=args.eval_id)
            eval_run = SimpleNamespace(id=args.run_id)
            write_in_progress(eval_object.id, eval_run.id)
        else:
            data_id = upload_dataset()
            eval_object = create_evaluation_definition()
            eval_run = run_evaluation(eval_object, data_id)
            write_in_progress(eval_object.id, eval_run.id)

        run = poll_for_results(eval_object, eval_run)
        retrieve_and_display_results(eval_object, run)

        section("Cloud evaluation complete")
        print(f"\nNext steps:")
        print(f"  1. Review detailed results in Azure AI Foundry portal")
        print(f"  2. Analyze patterns in successful and failed evaluations")
        print(f"  3. Commit {RESULTS_FILE} and push so the PR workflow can use it")

    except KeyboardInterrupt:
        if eval_object and eval_run:
            write_in_progress(eval_object.id, eval_run.id)
            print(
                "\nInterrupted. Cloud evaluation is still running.\n"
                f"Resume with:\n  {resume_command(eval_object.id, eval_run.id)}"
            )
        sys.exit(130)
    except Exception as e:
        error_message = (
            f"{'=' * 80}\n"
            f" Trail Guide Agent - Evaluation FAILED\n"
            f"{'=' * 80}\n"
            f"\nError: {e}\n"
            f"\nTroubleshooting:\n"
            f"  - Verify AZURE_AI_PROJECT_ENDPOINT in .env file\n"
            f"  - Check Azure credentials: az login\n"
            f"  - Ensure GPT-5.1 model is deployed and accessible\n"
            f"  - Ensure the caller has Foundry User access at the AI account scope\n"
            f"  - If you just ran azd up, wait 1-2 minutes for role propagation and retry once\n"
            f"  - If a Run ID was printed, resume polling instead of starting a new evaluation\n"
        )
        print(error_message)
        # Do not overwrite an in-progress marker with FAILED while we still
        # have live IDs — the cloud job may still be scoring.
        if eval_object and eval_run and "still be scoring" in str(e):
            write_in_progress(eval_object.id, eval_run.id)
        else:
            RESULTS_FILE.write_text(error_message, encoding="utf-8")
        sys.exit(1)


if __name__ == "__main__":
    main()
