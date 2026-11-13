"""Hugging Face Hub login that works the same on a laptop, Colab and Kaggle."""

import os

DEFAULT_SECRET_NAME = "hg-main"


def login_to_hub(secret_name: str = DEFAULT_SECRET_NAME) -> None:
    """Log in to the Hugging Face Hub with the first token found.

    Looks in order at: the HF_TOKEN environment variable, the Colab secret
    `secret_name`, the Kaggle secret `secret_name`, and finally an existing
    `huggingface-cli login`. Raises RuntimeError only if none of them has a token.
    """
    from huggingface_hub import get_token, login

    token = os.environ.get("HF_TOKEN") or _colab_secret(secret_name) or _kaggle_secret(secret_name)
    if token:
        login(token=token, add_to_git_credential=False)
        return
    if get_token():
        return
    raise RuntimeError(
        f"No Hugging Face token found. Set HF_TOKEN, add a '{secret_name}' secret "
        "on Colab/Kaggle, or run `huggingface-cli login`."
    )


def _colab_secret(name: str):
    try:
        from google.colab import userdata
        return userdata.get(name)
    except Exception:  # not on Colab, or the secret is missing
        return None


def _kaggle_secret(name: str):
    try:
        from kaggle_secrets import UserSecretsClient
        return UserSecretsClient().get_secret(name)
    except Exception:  # not on Kaggle, or the secret is missing
        return None
