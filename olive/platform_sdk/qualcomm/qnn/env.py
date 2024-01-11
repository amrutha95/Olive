# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import os
import platform
from pathlib import Path

from olive.platform_sdk.qualcomm.env import SDKEnv


class QNNSDKEnv(SDKEnv):
    def __init__(self, target_arch: str = None, dev: bool = False):
        super().__init__("QNN", "QNN_SDK_ROOT", target_arch=target_arch, dev=dev)

    @property
    def env(self):
        env = super().env
        sdk_root_path = self.sdk_root_path
        delimiter = os.path.pathsep
        python_env_bin_path = str(Path(f"{sdk_root_path}/olive-pyenv/bin"))
        if platform.system() == "Linux":
            if self.dev:
                if not Path(python_env_bin_path).exists():
                    raise FileNotFoundError(
                        f"Path {python_env_bin_path} does not exist. Please run"
                        " 'python -m olive.platform_sdk.qualcomm.qnn.configure --py_version 3.8'"
                        " to add the missing file."
                    )
                env["PATH"] = python_env_bin_path + delimiter + env["PATH"]
        env["QNN_SDK_ROOT"] = sdk_root_path
        return env

    def get_qnn_backend(self, backend_name):
        backend_path = Path(self.sdk_root_path) / "lib" / self.target_arch / backend_name
        if not backend_path.exists():
            raise FileNotFoundError(f"QNN backend {backend_path} does not exist.")
        return backend_path
