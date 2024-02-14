# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
# Export a PyTorch model using the onnxruntime-genai package.
# --------------------------------------------------------------------------
import logging
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict

from onnx import TensorProto

from olive.common.pydantic_v1 import validator
from olive.hardware.accelerator import AcceleratorLookup, AcceleratorSpec, Device
from olive.model import ONNXModelHandler, PyTorchModelHandler
from olive.model.utils import resolve_onnx_path
from olive.passes import Pass
from olive.passes.olive_pass import PassConfigParam

logger = logging.getLogger(__name__)


class GenAIModelExporter(Pass):
    class Precision(str, Enum):
        FP32 = "fp32"
        FP16 = "fp16"
        INT4 = "int4"

    @staticmethod
    def _default_config(accelerator_spec: AcceleratorSpec) -> Dict[str, PassConfigParam]:
        return {
            "precision": PassConfigParam(
                type_=str,
                required=True,
                description="Precision of model.",
            )
        }

    @staticmethod
    def _validate_precision(v):
        valid_precisions = [d.value for d in GenAIModelExporter.Precision]
        if v not in valid_precisions:
            raise ValueError(f"Invalid precision: {v}. Valid values are: {valid_precisions}")
        return v

    @staticmethod
    def _validators() -> Dict[str, Callable]:
        return {
            "validate_precision": validator("precision", allow_reuse=True)(GenAIModelExporter._validate_precision),
        }

    def _run_for_config(
        self, model: PyTorchModelHandler, data_root: str, config: Dict[str, Any], output_model_path: str
    ) -> ONNXModelHandler:
        from argparse import Namespace

        from olive.passes.pytorch.genai_exporter import create_model

        output_model_path = Path(output_model_path)
        output_model_path.mkdir(parents=True, exist_ok=True)
        output_model_path = resolve_onnx_path(output_model_path)

        precision = config["precision"]
        device = (
            Device.CPU
            if self.accelerator_spec.execution_provider in AcceleratorLookup.EXECUTION_PROVIDERS["cpu"]
            else Device.GPU
        )

        logger.info(
            "Valid precision + execution provider combinations are: FP32 CPU, FP32 CUDA, FP16 CUDA, INT4 CPU, INT4 CUDA"
        )

        # Set input/output precision of ONNX model
        io_dtype = (
            TensorProto.FLOAT
            if precision in {"int8", "fp32"} or (precision == "int4" and device == Device.CPU)
            else TensorProto.FLOAT16
        )

        args = Namespace(
            model_name_or_path=str(model.hf_config.model_name),
            io_dtype=io_dtype,
            output=str(output_model_path),
            precision=str(precision),
            execution_provider=str(device),
            cache_dir=str(Path.cwd() / "cache_dir"),
        )
        create_model(args)
        return ONNXModelHandler(output_model_path)
