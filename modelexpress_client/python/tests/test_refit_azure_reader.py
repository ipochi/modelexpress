# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

pytest.importorskip("azure.storage.blob")
pytest.importorskip("azure.identity")

from azure.core.exceptions import (
    HttpResponseError,
    ResourceNotFoundError,
    ServiceResponseError,
)
import azure.identity
import azure.storage.blob

from modelexpress_rl.azure import AzureBlobReader
from modelexpress_rl.object_storage_reader import ObjectStorageReader


class _BlobService:
    def __init__(self, objects):
        self.objects = objects
        self.calls = []
        self.error = None
        self.close = Mock()

    def get_blob_client(self, *, container, blob):
        key = (container, blob)

        def read(operation):
            self.calls.append((operation, key))
            if self.error is not None:
                raise self.error
            if key not in self.objects:
                raise ResourceNotFoundError("BlobNotFound")
            return self.objects[key]

        return SimpleNamespace(
            download_blob=lambda: SimpleNamespace(readall=lambda: read("get")),
            get_blob_properties=lambda: SimpleNamespace(size=len(read("size"))),
        )


@pytest.fixture
def sdk(monkeypatch):
    monkeypatch.delenv("AZURE_STORAGE_ACCOUNT_NAME", raising=False)
    monkeypatch.delenv("AZURE_STORAGE_CONNECTION_STRING", raising=False)
    client = _BlobService({("models", "v2/shard.safetensors"): b"weights"})
    factory = Mock(return_value=client)
    factory.from_connection_string.return_value = client
    credential = Mock()
    credential_factory = Mock(return_value=credential)
    monkeypatch.setattr(azure.storage.blob, "BlobServiceClient", factory)
    monkeypatch.setattr(azure.identity, "DefaultAzureCredential", credential_factory)
    return SimpleNamespace(
        client=client,
        factory=factory,
        credential=credential,
        credential_factory=credential_factory,
    )


@pytest.mark.parametrize("data", [b"", b"checkpoint\x00\xff"])
@pytest.mark.parametrize(
    "blob", ["model.safetensors.index.json", "v2/shard.safetensors", "v2/weights%20x"]
)
def test_reader_returns_exact_object_bytes_and_size(blob, data):
    client = _BlobService({("models", blob): data})
    reader: ObjectStorageReader = AzureBlobReader(client=client)

    assert reader.get(f"az://models/{blob}") == data
    assert reader.size(f"az://models/{blob}") == len(data)
    assert client.calls == [("get", ("models", blob)), ("size", ("models", blob))]


def test_size_does_not_download_blob():
    client = _BlobService({("models", "shard"): b"weights"})
    reader = AzureBlobReader(client=client)

    assert reader.size("az://models/shard") == 7
    assert client.calls == [("size", ("models", "shard"))]


@pytest.mark.parametrize("operation", ["get", "size"])
@pytest.mark.parametrize(
    "uri",
    [
        "",
        "models/shard",
        "s3://models/shard",
        "https://account.blob.core.windows.net/models/shard",
        "AZ://models/shard",
        "az:///shard",
        "az://models",
        "az://models/",
        "az://models//shard",
        "az://models/shard?sig=secret",
        "az://models/shard?",
        "az://models/shard#fragment",
        "az://models/shard#",
        "az://user@models/shard",
        "az://models:443/shard",
        "az://[models/shard",
        "az://mod els/shard",
        "az://models%2fother/shard",
        "az://models\\other/shard",
        " az://models/shard",
        "az://models/shard\n",
        "az://models/sh\tard",
    ],
)
def test_malformed_uri_fails_before_storage_access(uri, operation):
    client = Mock()
    reader = AzureBlobReader(client=client)

    with pytest.raises(ValueError, match="invalid Azure Blob URI") as error:
        getattr(reader, operation)(uri)

    client.get_blob_client.assert_not_called()
    assert "secret" not in str(error.value)


@pytest.mark.parametrize("operation", ["get", "size"])
def test_missing_blob_preserves_not_found_error(operation):
    reader = AzureBlobReader(client=_BlobService({}))

    with pytest.raises(ResourceNotFoundError):
        getattr(reader, operation)("az://models/missing")


@pytest.mark.parametrize("operation", ["get", "size"])
def test_authorization_failure_is_not_treated_as_missing_or_empty(operation):
    client = _BlobService({})
    client.error = HttpResponseError("AuthorizationPermissionMismatch")
    reader = AzureBlobReader(client=client)

    with pytest.raises(HttpResponseError) as error:
        getattr(reader, operation)("az://models/shard")

    assert error.value is client.error


def test_interrupted_download_does_not_return_partial_contents():
    client = _BlobService({("models", "shard"): b"weights"})
    client.error = ServiceResponseError("connection interrupted during readall")
    reader = AzureBlobReader(client=client)

    with pytest.raises(ServiceResponseError) as error:
        reader.get("az://models/shard")

    assert error.value is client.error


def test_one_reader_handles_concurrent_reads_from_multiple_containers():
    objects = {(f"container{i}", "shard"): bytes([i]) for i in range(16)}
    reader = AzureBlobReader(client=_BlobService(objects))
    uris = [f"az://{container}/{blob}" for container, blob in objects]

    with ThreadPoolExecutor(max_workers=4) as pool:
        assert list(pool.map(reader.get, uris)) == list(objects.values())


def test_connection_string_takes_precedence_over_account_name(sdk, monkeypatch):
    monkeypatch.setenv("AZURE_STORAGE_CONNECTION_STRING", "test-connection-string")
    monkeypatch.setenv("AZURE_STORAGE_ACCOUNT_NAME", "unusedaccount")

    reader = AzureBlobReader()

    sdk.factory.from_connection_string.assert_called_once_with("test-connection-string")
    sdk.factory.assert_not_called()
    sdk.credential_factory.assert_not_called()
    assert reader.get("az://models/v2/shard.safetensors") == b"weights"
    reader.close()
    reader.close()
    sdk.client.close.assert_called_once_with()
    sdk.credential.close.assert_not_called()


def test_account_name_uses_default_credential_and_closes_owned_resources(sdk, monkeypatch):
    monkeypatch.setenv("AZURE_STORAGE_ACCOUNT_NAME", " testaccount ")
    monkeypatch.setenv("AZURE_STORAGE_CONNECTION_STRING", " ")

    reader = AzureBlobReader()

    sdk.credential_factory.assert_called_once_with()
    sdk.factory.assert_called_once_with(
        account_url="https://testaccount.blob.core.windows.net",
        credential=sdk.credential,
    )
    sdk.factory.from_connection_string.assert_not_called()
    assert reader.size("az://models/v2/shard.safetensors") == len(b"weights")
    reader.close()
    reader.close()
    sdk.client.close.assert_called_once_with()
    sdk.credential.close.assert_called_once_with()


@pytest.mark.parametrize("account_name", [None, "", " "])
def test_missing_account_configuration_fails_before_credential_creation(
    sdk, monkeypatch, account_name
):
    if account_name is not None:
        monkeypatch.setenv("AZURE_STORAGE_ACCOUNT_NAME", account_name)

    with pytest.raises(ValueError, match="AZURE_STORAGE_ACCOUNT_NAME is required"):
        AzureBlobReader()

    sdk.factory.assert_not_called()
    sdk.credential_factory.assert_not_called()


def test_invalid_connection_string_does_not_fall_back_to_identity(sdk, monkeypatch):
    monkeypatch.setenv("AZURE_STORAGE_CONNECTION_STRING", "invalid")
    monkeypatch.setenv("AZURE_STORAGE_ACCOUNT_NAME", "testaccount")
    sdk.factory.from_connection_string.side_effect = ValueError("invalid connection string")

    with pytest.raises(ValueError, match="invalid connection string"):
        AzureBlobReader()

    sdk.factory.assert_not_called()
    sdk.credential_factory.assert_not_called()


def test_injected_client_needs_no_environment_and_remains_caller_owned(sdk):
    reader = AzureBlobReader(client=sdk.client)

    assert reader.get("az://models/v2/shard.safetensors") == b"weights"
    reader.close()

    sdk.factory.assert_not_called()
    sdk.credential_factory.assert_not_called()
    sdk.client.close.assert_not_called()


def test_client_construction_failure_closes_credential(sdk, monkeypatch):
    monkeypatch.setenv("AZURE_STORAGE_ACCOUNT_NAME", "testaccount")
    sdk.factory.side_effect = ValueError("client construction failed")

    with pytest.raises(ValueError, match="client construction failed"):
        AzureBlobReader()

    sdk.credential.close.assert_called_once_with()


def test_client_close_failure_still_closes_credential(sdk, monkeypatch):
    monkeypatch.setenv("AZURE_STORAGE_ACCOUNT_NAME", "testaccount")
    sdk.client.close.side_effect = OSError("client close failed")
    reader = AzureBlobReader()

    with pytest.raises(OSError, match="client close failed"):
        reader.close()
    reader.close()

    sdk.client.close.assert_called_once_with()
    sdk.credential.close.assert_called_once_with()
