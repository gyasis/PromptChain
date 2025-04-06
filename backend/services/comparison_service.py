from typing import Dict, Any, Callable, Awaitable
import asyncio
import time
from ...examples.chain_of_draft_comparison import ChainOfDraftComparison

class ComparisonService:
    def __init__(self):
        self.comparison = ChainOfDraftComparison()

    async def run_comparison(
        self,
        problem: str,
        model: str,
        progress_callback: Callable[[str, float], Awaitable[None]]
    ) -> Dict[str, Any]:
        """
        Run a comparison of different Chain of Draft approaches.
        
        Args:
            problem: The problem to solve
            model: The model to use
            progress_callback: Async callback for progress updates
        
        Returns:
            Dict containing comparison results
        """
        start_time = time.time()
        
        # Track progress for each mode
        progress = {
            'batch': 0.0,
            'sequential': 0.0,
            'mirror': 0.0
        }

        async def update_progress(mode: str, value: float):
            progress[mode] = value
            await progress_callback(mode, value)

        # Run batch mode
        batch_result = await self._run_batch_mode(
            problem, model,
            lambda p: asyncio.create_task(update_progress('batch', p))
        )
        
        # Run sequential mode
        sequential_result = await self._run_sequential_mode(
            problem, model,
            lambda p: asyncio.create_task(update_progress('sequential', p))
        )
        
        # Run mirror mode
        mirror_result = await self._run_mirror_mode(
            problem, model,
            lambda p: asyncio.create_task(update_progress('mirror', p))
        )

        # Calculate execution times
        end_time = time.time()
        total_time = (end_time - start_time) * 1000  # Convert to milliseconds

        # Combine results
        return {
            'batch': {
                **batch_result,
                'execution_time': total_time / 3  # Approximate time per mode
            },
            'sequential': {
                **sequential_result,
                'execution_time': total_time / 3
            },
            'mirror': {
                **mirror_result,
                'execution_time': total_time / 3
            },
            'total_execution_time': total_time
        }

    async def _run_batch_mode(
        self,
        problem: str,
        model: str,
        progress_callback: Callable[[float], None]
    ) -> Dict[str, Any]:
        """Run comparison in batch mode."""
        progress_callback(0.1)  # Starting
        
        result = await self.comparison.solve_batch(
            problem=problem,
            model=model
        )
        
        progress_callback(1.0)  # Complete
        return result

    async def _run_sequential_mode(
        self,
        problem: str,
        model: str,
        progress_callback: Callable[[float], None]
    ) -> Dict[str, Any]:
        """Run comparison in sequential mode."""
        progress_callback(0.1)  # Starting
        
        result = await self.comparison.solve_sequential(
            problem=problem,
            model=model
        )
        
        progress_callback(1.0)  # Complete
        return result

    async def _run_mirror_mode(
        self,
        problem: str,
        model: str,
        progress_callback: Callable[[float], None]
    ) -> Dict[str, Any]:
        """Run comparison in mirror mode."""
        progress_callback(0.1)  # Starting
        
        result = await self.comparison.solve_mirror(
            problem=problem,
            model=model
        )
        
        progress_callback(1.0)  # Complete
        return result 