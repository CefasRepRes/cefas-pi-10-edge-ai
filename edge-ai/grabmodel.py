# grabmodel.py
"""Download trained Edge-AI model artefacts from the trainedmodels blob container.

Usage:
    python grabmodel.py --sas-token <token>
    python grabmodel.py --sas-token <token> model2026-07-21T09-12-07Z
    python grabmodel.py --sas-token <token> model2026-07-21T09-12-07Z.pt

    # The leading '?' is optional; both of these are equivalent:
    python grabmodel.py --sas-token "?sv=2025-..."
    python grabmodel.py --sas-token "sv=2025-..."

    # If --sas-token is omitted the script falls back to an interactive prompt.

For the selected model this downloads, where present:
    models/model<timestamp>.pt
    models/training_report<timestamp>.json
    models/modeltrainsettings.json
"""

from pathlib import Path
import argparse
import sys

from azure.storage.blob import ContainerClient
from azure.core.exceptions import ResourceNotFoundError

ACCOUNT_NAME = "citprodc8603uksa"
CONTAINER_NAME = "trainedmodels"
OUTPUT_DIR = Path("models")


def normalise_model_name(model_name: str) -> str:
    model_name = Path(str(model_name).strip()).name
    if not model_name:
        raise ValueError("No model name supplied")
    if not model_name.endswith(".pt"):
        model_name = f"{model_name}.pt"
    if not model_name.startswith("model"):
        raise ValueError("Model name should look like model2026-07-21T09-12-07Z or model2026-07-21T09-12-07Z.pt")
    return model_name


def model_timestamp(model_name: str) -> str:
    model_name = normalise_model_name(model_name)
    return model_name[len("model"):-len(".pt")]


def matching_training_report(model_name: str) -> str:
    return f"training_report{model_timestamp(model_name)}.json"


def normalise_sas_token(sas: str) -> str:
    """Ensure the SAS token starts with '?'."""
    sas = sas.strip()
    if not sas.startswith("?"):
        sas = "?" + sas
    return sas


def get_container_client(sas_token: str = None):
    """Return a ContainerClient authenticated with the given SAS token.

    If *sas_token* is not provided (or is empty) the user is prompted
    interactively via stdin.  Pass ``--sas-token`` on the command line to
    avoid the interactive prompt — useful when the script is launched as a
    subprocess and stdin is not a terminal.
    """
    if sas_token:
        sas = normalise_sas_token(sas_token)
    else:
        print(
            "\nPaste an SAS token to "
            f"https://{ACCOUNT_NAME}.blob.core.windows.net/{CONTAINER_NAME} "
            "(sorry but IT have blocked authentication on non-Cefas devices)."
        )
        print("Example:")
        print("?sv=2025-...")

        sas = input("\nSAS token: ").strip()
        if not sas:
            raise RuntimeError("No SAS token supplied")
        sas = normalise_sas_token(sas)

    container_url = f"https://{ACCOUNT_NAME}.blob.core.windows.net/{CONTAINER_NAME}{sas}"
    return ContainerClient.from_container_url(container_url)


def list_models(container_client):
    models = []
    print("\nFetching available models...\n")
    for blob in container_client.list_blobs():
        name = blob.name
        basename = Path(name).name
        if basename.startswith("model") and basename.endswith(".pt"):
            models.append(name)
    models.sort(reverse=True)
    return models


def choose_model(models):
    print("Available models:\n")
    for i, model in enumerate(models, start=1):
        print(f"{i:3d}: {model}")

    while True:
        try:
            choice = int(input("\nSelect model number: "))
            if 1 <= choice <= len(models):
                return models[choice - 1]
        except ValueError:
            pass
        print("Invalid selection.")


def download_blob(container_client, blob_name, *, required=True):
    OUTPUT_DIR.mkdir(exist_ok=True)
    destination = OUTPUT_DIR / Path(blob_name).name

    if destination.exists() and destination.stat().st_size > 0:
        print(f"Already present: {destination}")
        return destination

    print(f"\nDownloading {blob_name}")
    blob_client = container_client.get_blob_client(blob_name)

    try:
        with open(destination, "wb") as f:
            f.write(blob_client.download_blob().readall())
    except ResourceNotFoundError:
        if required:
            raise
        print(f"Optional file not found: {blob_name}")
        return None

    print(f"Saved: {destination}")
    return destination


def download_model_bundle(container_client, model_name):
    model_blob = normalise_model_name(model_name)
    report_blob = matching_training_report(model_blob)

    print("\nSelected:")
    print(f"  Model          : {model_blob}")
    print(f"  Training report: {report_blob}")

    return {
        "model": download_blob(container_client, model_blob, required=True),
        "training_report": download_blob(container_client, report_blob, required=True),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description="Download Edge-AI model artefacts from Azure Blob Storage")
    parser.add_argument(
        "model",
        nargs="?",
        help="model<timestamp> or model<timestamp>.pt. If omitted, choose from a numbered list.",
    )
    parser.add_argument(
        "--sas-token",
        metavar="TOKEN",
        help=(
            "SAS token for the Azure Blob Storage container. "
            "The leading '?' is optional. "
            "If omitted, the script will prompt interactively."
        ),
    )
    args = parser.parse_args(argv)

    try:
        container_client = get_container_client(args.sas_token)
        if args.model:
            selected_model = normalise_model_name(args.model)
        else:
            models = list_models(container_client)
            if not models:
                print("No models found.")
                return 1
            selected_model = choose_model(models)

        download_model_bundle(container_client, selected_model)
        print("\nFinished.")
        return 0

    except KeyboardInterrupt:
        print("\nCancelled.")
        return 130
    except Exception as e:
        print(f"\nERROR: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
