"""
Dask configuration and client management utilities.
"""
import dask
import dask.config
from dask.distributed import Client
from .logger import LoggingPlugin, get_logger
from .config_settings import DASK_TEMP_DIR, CPU_LIMIT, RAM_LIMIT


class DaskManager:
    """Manages Dask configuration and client lifecycle"""
    
    def __init__(self):
        """Initialize DaskManager with no active client"""
        self.client = None
        self.logger = get_logger(__name__)
        
    def setup_dask_config(self):
        """Configure Dask settings"""
        dask.config.set({
            'distributed.worker.daemon': False,  # Run worker as daemon
            'distributed.shuffle.method': "tasks",  # Use tasks instead of p2p for shuffle
            'distributed.worker.memory.target': 0.9,  # Target memory usage
            'distributed.worker.memory.spill': 0.6,   # Spill threshold
            'distributed.worker.memory.pause': 0.8,  # Pause threshold
            'distributed.worker.memory.terminate': 0.95,  # Terminate threshold
            'temporary_directory': DASK_TEMP_DIR,
            # [NEW] Timeout settings to handle long tasks
            'distributed.nanny.worker_process_timeout': 180,  # Increase timeout
            'distributed.nanny.pre_spawn_environ_timeout': 180,  # Increase timeout
            'distributed.comm.timeouts.connect': 180,  # Connection timeout
            'distributed.comm.timeouts.tcp': 180,  # TCP timeout
            'distributed.worker.lifetime.duration': None,  # Disable worker cycling
            'distributed.worker.lifetime.stagger': None,  # Disable staggered restarts
        })
        self.logger.info("Dask configuration applied")
        
    def create_client(self):
        """
        Initialize and configure Dask client.
        
        Returns:
            Client: Configured Dask client
        """
        self.client = Client(
            n_workers=CPU_LIMIT, 
            threads_per_worker=2, 
            memory_limit=f"{float(RAM_LIMIT / CPU_LIMIT)}GB",
            local_directory=DASK_TEMP_DIR, #"/tmp",
            dashboard_address=":8787",
            timeout="180s",  # Increase general timeout
            death_timeout="180s",  # Increase death timeout
        )
        
        self.logger.info(f"Dask Client started: {self.client}")
        self.logger.info(f"Dashboard: {self.client.dashboard_link}")
        self.logger.info(f"Number of workers = {CPU_LIMIT}")
        self.logger.info(f"Threads per worker = 2")
        self.logger.info(f"Memory limit = {float(RAM_LIMIT / CPU_LIMIT)}GB per worker")

        # Register plugin for worker logging
        plugin = LoggingPlugin()
        self.client.register_plugin(plugin)
        
        return self.client
        
    def wait_for_workers(self):
        """Ensure all workers are ready"""
        if self.client:
            self.client.wait_for_workers(n_workers=CPU_LIMIT, timeout=180)
            self.logger.info("All Dask workers are ready")
        else:
            self.logger.warning("No Dask client available to wait for workers")
            
    def close(self):
        """Close the Dask client and cleanup resources"""
        if self.client:
            try:
                # Don't cancel tasks - let them complete naturally
                
                # Close with extended timeout
                self.client.close(timeout=30)  # Reduced from 180 to 30
                self.logger.info("Dask client closed")
            except Exception as e:
                self.logger.warning(f"Error during client close (may be harmless): {e}")
                # Force close if graceful close fails
                try:
                    self.client.close(timeout=5)  # Quick force close
                except:
                    pass
            finally:
                self.client = None
        else:
            self.logger.warning("No Dask client to close")
            
    def __enter__(self):
        """Context manager entry - setup and create client"""
        self.setup_dask_config()
        self.create_client()
        self.wait_for_workers()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleanup client"""
        self.close()
        # Suppress the timeout exception if it occurs during cleanup
        if exc_type is TimeoutError:
            self.logger.warning("Timeout during Dask cleanup - this is usually harmless")
            return True  # Suppress the exception
        return False