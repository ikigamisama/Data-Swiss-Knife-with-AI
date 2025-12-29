import pandas as pd
from typing import List, Dict, Any, Optional, Callable
from datetime import datetime
from core.base import DataProcessor, DataContext
from core.exceptions import PipelineError
import logging
import json

logger = logging.getLogger(__name__)


class ETLPipeline:
    """ETL Pipeline using Chain of Responsibility pattern"""

    def __init__(self, name: str = "Unnamed Pipeline"):
        """
        Initialize ETL pipeline

        Args:
            name: Pipeline name
        """
        self.name = name
        self.steps: List[DataProcessor] = []
        self.metadata: Dict[str, Any] = {
            'created_at': datetime.now().isoformat(),
            'name': name
        }
        self.execution_log: List[Dict[str, Any]] = []

    def add_step(self, processor: DataProcessor, name: Optional[str] = None) -> 'ETLPipeline':
        """
        Add a processing step to the pipeline

        Args:
            processor: Data processor to add
            name: Step name (optional)

        Returns:
            Self for method chaining
        """
        self.steps.append({
            'processor': processor,
            'name': name or processor.__class__.__name__,
            'added_at': datetime.now().isoformat()
        })

        # Chain processors
        if len(self.steps) > 1:
            self.steps[-2]['processor'].set_next(processor)

        logger.info(f"Added step: {name or processor.__class__.__name__}")
        return self

    def execute(self, data: pd.DataFrame, validate: bool = True) -> DataContext:
        """
        Execute the pipeline

        Args:
            data: Input DataFrame
            validate: Whether to validate each step

        Returns:
            DataContext with processed data
        """
        try:
            # Create initial context
            context = DataContext(
                data=data,
                metadata={
                    'pipeline_name': self.name,
                    'start_time': datetime.now().isoformat(),
                    'input_shape': data.shape
                },
                source='pipeline',
                timestamp=datetime.now().isoformat()
            )

            logger.info(f"Starting pipeline: {self.name}")
            logger.info(f"Input shape: {data.shape}")

            # Execute first step (chain will handle rest)
            if self.steps:
                context = self.steps[0]['processor'].process(context)

            # Add final metadata
            context.metadata['end_time'] = datetime.now().isoformat()
            context.metadata['output_shape'] = context.data.shape
            context.metadata['steps_executed'] = len(self.steps)

            logger.info(f"Pipeline completed: {self.name}")
            logger.info(f"Output shape: {context.data.shape}")

            return context

        except Exception as e:
            logger.error(f"Pipeline execution failed: {e}")
            raise PipelineError(f"Pipeline '{self.name}' failed: {str(e)}")

    def validate_pipeline(self) -> Dict[str, Any]:
        """Validate pipeline configuration"""
        validation = {
            'valid': True,
            'issues': [],
            'warnings': []
        }

        if not self.steps:
            validation['valid'] = False
            validation['issues'].append("Pipeline has no steps")

        # Check for incompatible step sequences
        # Add custom validation logic here

        return validation

    def get_summary(self) -> Dict[str, Any]:
        """Get pipeline summary"""
        return {
            'name': self.name,
            'steps': len(self.steps),
            'step_names': [step['name'] for step in self.steps],
            'metadata': self.metadata,
            'created_at': self.metadata.get('created_at')
        }

    def save_config(self, filepath: str):
        """Save pipeline configuration to file"""
        config = {
            'name': self.name,
            'metadata': self.metadata,
            'steps': [
                {
                    'name': step['name'],
                    'type': step['processor'].__class__.__name__,
                    'added_at': step['added_at']
                }
                for step in self.steps
            ]
        }

        with open(filepath, 'w') as f:
            json.dump(config, f, indent=2)

        logger.info(f"Pipeline config saved to {filepath}")

    @classmethod
    def load_config(cls, filepath: str) -> 'ETLPipeline':
        """Load pipeline configuration from file"""
        with open(filepath, 'r') as f:
            config = json.load(f)

        pipeline = cls(name=config['name'])
        pipeline.metadata = config.get('metadata', {})

        logger.info(f"Pipeline config loaded from {filepath}")
        return pipeline

    def clear(self):
        """Clear all steps from pipeline"""
        self.steps.clear()
        self.execution_log.clear()
        logger.info("Pipeline cleared")


class PipelineBuilder:
    """Builder for constructing complex pipelines"""

    def __init__(self, name: str = "Pipeline"):
        self.pipeline = ETLPipeline(name=name)
        self._current_step = None

    def with_name(self, name: str) -> 'PipelineBuilder':
        """Set pipeline name"""
        self.pipeline.name = name
        return self

    def add_step(self, processor: DataProcessor, name: Optional[str] = None) -> 'PipelineBuilder':
        """Add processing step"""
        self.pipeline.add_step(processor, name)
        return self

    def add_metadata(self, key: str, value: Any) -> 'PipelineBuilder':
        """Add metadata to pipeline"""
        self.pipeline.metadata[key] = value
        return self

    def build(self) -> ETLPipeline:
        """Build and return the pipeline"""
        return self.pipeline


class ParallelPipeline:
    """Execute multiple pipelines in parallel"""

    def __init__(self, pipelines: List[ETLPipeline]):
        """
        Initialize parallel pipeline

        Args:
            pipelines: List of pipelines to execute
        """
        self.pipelines = pipelines

    def execute(self, data: pd.DataFrame) -> Dict[str, DataContext]:
        """
        Execute all pipelines

        Args:
            data: Input DataFrame

        Returns:
            Dictionary mapping pipeline names to results
        """
        results = {}

        for pipeline in self.pipelines:
            try:
                result = pipeline.execute(data.copy())
                results[pipeline.name] = result
                logger.info(f"Pipeline '{pipeline.name}' completed")
            except Exception as e:
                logger.error(f"Pipeline '{pipeline.name}' failed: {e}")
                results[pipeline.name] = {'error': str(e)}

        return results


class ConditionalPipeline(ETLPipeline):
    """Pipeline with conditional execution"""

    def __init__(self, name: str = "Conditional Pipeline"):
        super().__init__(name)
        self.conditions: List[Callable] = []

    def add_conditional_step(self, processor: DataProcessor,
                             condition: Callable[[pd.DataFrame], bool],
                             name: Optional[str] = None) -> 'ConditionalPipeline':
        """
        Add a step that executes only if condition is met

        Args:
            processor: Data processor
            condition: Function that returns True/False
            name: Step name

        Returns:
            Self for chaining
        """
        self.add_step(processor, name)
        self.conditions.append(condition)
        return self

    def execute(self, data: pd.DataFrame, validate: bool = True) -> DataContext:
        """Execute pipeline with conditions"""
        context = DataContext(
            data=data,
            metadata={'pipeline_name': self.name},
            source='conditional_pipeline',
            timestamp=datetime.now().isoformat()
        )

        for idx, step in enumerate(self.steps):
            # Check condition if exists
            if idx < len(self.conditions):
                if not self.conditions[idx](context.data):
                    logger.info(
                        f"Skipping step {step['name']} (condition not met)")
                    continue

            # Execute step
            context = step['processor'].process(context)

        return context


class PipelineMonitor:
    """Monitor pipeline execution"""

    def __init__(self):
        self.executions: List[Dict[str, Any]] = []

    def log_execution(self, pipeline_name: str, context: DataContext):
        """Log a pipeline execution"""
        execution = {
            'pipeline': pipeline_name,
            'timestamp': datetime.now().isoformat(),
            'input_shape': context.metadata.get('input_shape'),
            'output_shape': context.metadata.get('output_shape'),
            'duration': self._calculate_duration(context.metadata),
            'success': True
        }

        self.executions.append(execution)

    def log_failure(self, pipeline_name: str, error: str):
        """Log a pipeline failure"""
        execution = {
            'pipeline': pipeline_name,
            'timestamp': datetime.now().isoformat(),
            'error': error,
            'success': False
        }

        self.executions.append(execution)

    def get_statistics(self) -> Dict[str, Any]:
        """Get execution statistics"""
        if not self.executions:
            return {}

        total = len(self.executions)
        successful = sum(1 for e in self.executions if e['success'])

        return {
            'total_executions': total,
            'successful': successful,
            'failed': total - successful,
            'success_rate': successful / total * 100
        }

    def _calculate_duration(self, metadata: Dict) -> Optional[float]:
        """Calculate execution duration"""
        try:
            start = datetime.fromisoformat(metadata.get('start_time', ''))
            end = datetime.fromisoformat(metadata.get('end_time', ''))
            return (end - start).total_seconds()
        except:
            return None
