from typing import Generator
from app.models.schemas import ProcessingConfig, SegmentationResult
from app.services.fastsurfer import FastSurferService
from app.services.file_handler import FileHandler
from app.services.preprocessing_service import Preprocessor

class ProcessingController:
    def __init__(self, config: dict):
        self.config = config
        self.fastsurfer = FastSurferService(config)
        self.file_handler = FileHandler(config)
        self.preprocessor = Preprocessor()

    def process_scan(self, config: ProcessingConfig) -> Generator[str, None, SegmentationResult]:
        try:
            # Setup directories
            temp_dir = self.file_handler.create_temp_dir()
            yield f"Created temporary directory: {temp_dir}"
            
            # Preprocessing
            yield "Preprocessing MRI scan..."
            preprocessed_path = self.preprocessor.resample_to_isotropic(config.input_path, temp_dir)
            
            # Segmentation
            yield "Running FastSurfer segmentation..."
            
            result = None
            for message in self.fastsurfer.segment(preprocessed_path, temp_dir):
                yield message  # Streaming messages during processing
            
            # Capture the final segmentation result
            result = self.fastsurfer._validate_output(temp_dir)

            # Post-processing
            yield "Finalizing results..."
            yield result
            
        except Exception as e:
            result = SegmentationResult(success=False, error=str(e))
            yield result  # Yield failure message
        
        finally:
            if config.cleanup:
                self.file_handler.cleanup(temp_dir)

        return result  # Ensure result is returned at the end
