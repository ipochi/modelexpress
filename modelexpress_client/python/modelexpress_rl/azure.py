# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Azure Blob reads for canonical checkpoint artifacts."""

from __future__ import annotations

from typing import TYPE_CHECKING
from urllib.parse import urlsplit

from modelexpress import envs

if TYPE_CHECKING:
    from azure.identity import DefaultAzureCredential
    from azure.storage.blob import BlobServiceClient


def _parse_uri(uri: str) -> tuple[str, str]:
    error = "invalid Azure Blob URI: expected az://container/blob without query or fragment"
    try:
        parsed = urlsplit(uri)
    except ValueError as exc:
        raise ValueError(error) from exc
    if (
        not uri.startswith("az://")
        or not parsed.netloc
        or parsed.netloc != parsed.hostname
        or any(char.isspace() for char in parsed.netloc)
        or any(char in parsed.netloc for char in "\\%")
        or not parsed.path.startswith("/")
        or parsed.path.startswith("//")
        or len(parsed.path) == 1
        or "?" in uri
        or "#" in uri
        or any(ord(char) < 32 or ord(char) == 127 for char in uri)
    ):
        raise ValueError(error)
    return parsed.netloc, parsed.path[1:]


class AzureBlobReader:
    """Read Blob objects with an injected client or environment credentials.

    An injected client remains owned by the caller. Otherwise, a connection
    string takes precedence over account name plus DefaultAzureCredential.
    """

    def __init__(self, *, client: BlobServiceClient | None = None) -> None:
        self._credential: DefaultAzureCredential | None = None
        self._owns_client = client is None
        self._closed = False
        if client is not None:
            self._client = client
            return

        try:
            from azure.storage.blob import BlobServiceClient
        except ImportError as exc:
            raise ImportError(
                "Azure Blob reads require the modelexpress[azure] extra"
            ) from exc

        connection_string = envs.AZURE_STORAGE_CONNECTION_STRING
        if connection_string:
            self._client = BlobServiceClient.from_connection_string(connection_string)
            return

        account_name = envs.AZURE_STORAGE_ACCOUNT_NAME
        if not account_name:
            raise ValueError(
                "AZURE_STORAGE_ACCOUNT_NAME is required when "
                "AZURE_STORAGE_CONNECTION_STRING is not set"
            )

        try:
            from azure.identity import DefaultAzureCredential
        except ImportError as exc:
            raise ImportError(
                "Azure Blob reads require the modelexpress[azure] extra"
            ) from exc

        self._credential = DefaultAzureCredential()
        try:
            self._client = BlobServiceClient(
                account_url=f"https://{account_name}.blob.core.windows.net",
                credential=self._credential,
            )
        except Exception:
            self._credential.close()
            raise

    def get(self, uri: str) -> bytes:
        container, blob = _parse_uri(uri)
        return self._client.get_blob_client(
            container=container, blob=blob
        ).download_blob().readall()

    def size(self, uri: str) -> int:
        container, blob = _parse_uri(uri)
        return self._client.get_blob_client(
            container=container, blob=blob
        ).get_blob_properties().size

    def close(self) -> None:
        if not self._owns_client or self._closed:
            return
        self._closed = True
        try:
            self._client.close()
        finally:
            if self._credential is not None:
                self._credential.close()


__all__ = ["AzureBlobReader"]
