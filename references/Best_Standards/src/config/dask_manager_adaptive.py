"""
Adaptive Dask configuration and client management utilities.
This version implements dynamic worker scaling based on workload and resource availability.
"""
import dask
import dask.config
from dask.distributed import Client, LocalCluster, Nanny
import threading
import time
from .logger import LoggingPlugin, get_logger
from .config_settings import DASK_TEMP_DIR, CPU_LIMIT, RAM_LIMIT


class AdaptiveDaskManager:
    """Manages Dask configuration and client lifecycle with adaptive scaling"""
    
    def __init__(self):
        """Initialize AdaptiveDaskManager with no active client"""
        self.client = None
        self.cluster = None
        self.logger = get_logger(__name__)
        self._last_worker_count = 0
        self._monitoring_thread = None
        self._stop_monitoring = False
        
    def setup_dask_config(self):
        """Configure Dask settings with conservative memory management for adaptive scaling"""
        dask.config.set({
            'distributed.worker.daemon': False,  # Run worker as daemon
            'distributed.shuffle.method': "tasks",  # Use tasks instead of p2p for shuffle
            # More conservative memory thresholds for adaptive scaling
            'distributed.worker.memory.target': 0.6,   # More conservative (was 0.8)
            'distributed.worker.memory.spill': 0.7,    # More conservative (was 0.9)  
            'distributed.worker.memory.pause': 0.8,    # More conservative (was 0.95)
            'distributed.worker.memory.terminate': 0.85, # More conservative (was 0.98)
            'temporary_directory': DASK_TEMP_DIR,
            # Enhanced memory monitoring for adaptive behavior
            'distributed.worker.memory.monitor-interval': '500ms',  # More frequent monitoring
            'distributed.worker.memory.rebalance.measure': 'optimistic',
            # Timeout settings for adaptive scaling
            'distributed.nanny.worker_process_timeout': 300,  # Longer timeout for adaptive workers
            'distributed.nanny.pre_spawn_environ_timeout': 300,
            'distributed.comm.timeouts.connect': 300,
            'distributed.comm.timeouts.tcp': 300,
            'distributed.worker.lifetime.duration': None,  # Disable worker cycling
            'distributed.worker.lifetime.stagger': None,  # Disable staggered restarts
            # Adaptive-specific settings
            'distributed.adaptive.interval': '2s',        # How often to check for scaling
            'distributed.adaptive.wait-count': 3,         # Wait periods before scaling down
            'distributed.adaptive.scale-factor': 2,       # How aggressively to scale
        })
        self.logger.info("Adaptive Dask configuration applied")
        
    def create_client(self):
        """
        Initialize and configure adaptive Dask client.
        
        Returns:
            Client: Configured adaptive Dask client
        """
        # Calculate base memory per worker
        # base_memory_per_worker = float(RAM_LIMIT / CPU_LIMIT)
        base_memory_per_worker = 4
        
        # Create LocalCluster for adaptive scaling
        self.cluster = LocalCluster(
            n_workers=0,  # Start with 0 workers - will scale up on demand
            threads_per_worker=2,
            memory_limit=f"{base_memory_per_worker}GB",
            local_directory=DASK_TEMP_DIR,  # Use configured temp directory
            dashboard_address=":8787",
            # Remove worker_class to use default (which uses Nanny automatically)
            silence_logs=False,  # Keep logging for monitoring
        )
        
        # Configure adaptive scaling behavior with basic parameters
        self.cluster.adapt(
            minimum=1,                    # Always keep at least 1 worker
            maximum=CPU_LIMIT,           # Respect CPU limit
            interval='2s',               # Check every 2 seconds for scaling decisions
            wait_count=3,                # Wait 3 intervals (6s) before scaling down
            target_duration='5s',        # Target 5 seconds of work per worker
        )
        
        # Create client from adaptive cluster
        self.client = Client(self.cluster)
        
        # Log adaptive configuration
        self.logger.info(f"Adaptive Dask Client started: {self.client}")
        self.logger.info(f"Dashboard: {self.client.dashboard_link}")
        self.logger.info(f"Adaptive scaling: min=1, max={CPU_LIMIT} workers")
        self.logger.info(f"Base memory per worker: {base_memory_per_worker}GB")
        self.logger.info(f"Total RAM available: {RAM_LIMIT}GB")
        
        # Register plugin for worker logging
        try:
            plugin = LoggingPlugin()
            self.client.register_plugin(plugin)
            self.logger.info("Logging plugin registered successfully")
        except Exception as e:
            self.logger.warning(f"Could not register logging plugin: {e}")
        
        # Start adaptive monitoring
        self.start_adaptive_monitoring()
        
        return self.client
        
    def get_scaling_status(self):
        """Get current adaptive scaling status"""
        if self.cluster and self.client:
            try:
                workers = self.client.scheduler_info()['workers']
                current_workers = len(workers)
                
                # Calculate effective memory per worker
                effective_memory_per_worker = f"{float(RAM_LIMIT / max(1, current_workers)):.1f}GB"
                
                # Get adaptive scaling information
                adaptive_info = {
                    'current_workers': current_workers,
                    'max_workers': CPU_LIMIT,
                    'effective_memory_per_worker': effective_memory_per_worker,
                    'total_memory_allocated': f"{current_workers * (RAM_LIMIT / max(1, current_workers)):.1f}GB",
                    'cluster_status': getattr(self.cluster, 'status', 'unknown'),
                }
                
                # Try to get adaptive target (may not always be available)
                if hasattr(self.cluster, 'adaptive') and self.cluster.adaptive:
                    try:
                        adaptive_info['target_workers'] = getattr(self.cluster.adaptive, 'target', 'calculating')
                        adaptive_info['pending_workers'] = getattr(self.cluster.adaptive, 'pending', 0)
                    except:
                        pass
                
                return adaptive_info
            except Exception as e:
                self.logger.warning(f"Could not get scaling status: {e}")
        return None

    def start_adaptive_monitoring(self):
        """Start monitoring thread for adaptive scaling behavior"""
        if self._monitoring_thread and self._monitoring_thread.is_alive():
            return  # Already monitoring
            
        self._stop_monitoring = False
        
        def monitor_loop():
            """Monitor and log adaptive scaling decisions"""
            self.logger.info("Adaptive scaling monitoring started")
            
            while not self._stop_monitoring and self.client and self.client.status != 'closed':
                try:
                    status = self.get_scaling_status()
                    if status:
                        current = status['current_workers']
                        
                        # Log significant scaling events
                        if current != self._last_worker_count:
                            self.logger.info(f"Worker count changed: {self._last_worker_count} -> {current}")
                            self.logger.info(f"New effective memory per worker: {status['effective_memory_per_worker']}")
                            self.logger.info(f"Total memory allocated: {status['total_memory_allocated']}")
                            
                            # Log memory efficiency gain/loss
                            if self._last_worker_count > 0:
                                old_memory = RAM_LIMIT / self._last_worker_count
                                new_memory = RAM_LIMIT / current if current > 0 else 0
                                if new_memory > old_memory:
                                    self.logger.info(f"Memory per worker increased by {((new_memory/old_memory - 1) * 100):.1f}%")
                                elif new_memory < old_memory and new_memory > 0:
                                    self.logger.info(f"Memory per worker decreased by {((1 - new_memory/old_memory) * 100):.1f}%")
                        
                        self._last_worker_count = current
                        
                        # Periodic status logging (every 30 seconds when stable)
                        if int(time.time()) % 30 == 0:
                            self.logger.debug(f"Adaptive status: {status}")
                    
                    time.sleep(2)  # Check every 2 seconds
                except Exception as e:
                    self.logger.error(f"Error in adaptive monitoring: {e}")
                    time.sleep(5)  # Wait longer on error
            
            self.logger.info("Adaptive scaling monitoring stopped")
        
        self._monitoring_thread = threading.Thread(target=monitor_loop, daemon=True)
        self._monitoring_thread.start()

    def stop_adaptive_monitoring(self):
        """Stop the adaptive monitoring thread"""
        self._stop_monitoring = True
        if self._monitoring_thread:
            self._monitoring_thread.join(timeout=5)
            
    def wait_for_workers(self, min_workers=1):
        """
        Wait for minimum number of workers to be ready.
        In adaptive mode, we only wait for the minimum.
        """
        if self.client:
            try:
                self.client.wait_for_workers(n_workers=min_workers, timeout=60)
                self.logger.info(f"At least {min_workers} Dask worker(s) ready")
                
                # Log initial scaling status
                status = self.get_scaling_status()
                if status:
                    self.logger.info(f"Initial scaling status: {status}")
            except Exception as e:
                self.logger.warning(f"Timeout waiting for workers (this may be normal in adaptive mode): {e}")
        else:
            self.logger.warning("No Dask client available to wait for workers")

    def force_scale_to(self, target_workers):
        """
        Manually force scaling to a specific number of workers.
        Useful for testing or specific workload requirements.
        """
        if self.cluster and 1 <= target_workers <= CPU_LIMIT:
            try:
                self.cluster.scale(target_workers)
                new_memory_per_worker = RAM_LIMIT / target_workers
                self.logger.info(f"Manually scaling to {target_workers} workers")
                self.logger.info(f"New memory per worker: {new_memory_per_worker:.1f}GB")
                return True
            except Exception as e:
                self.logger.error(f"Failed to manually scale to {target_workers} workers: {e}")
        else:
            self.logger.warning(f"Invalid target worker count: {target_workers} (must be 1-{CPU_LIMIT})")
        return False

    def get_memory_usage_summary(self):
        """Get detailed memory usage summary across all workers"""
        if not self.client:
            return None
            
        try:
            workers = self.client.scheduler_info()['workers']
            memory_summary = {
                'total_workers': len(workers),
                'total_memory_limit': 0,
                'total_memory_used': 0,
                'workers_detail': []
            }
            
            for worker_id, info in workers.items():
                worker_memory = {
                    'worker_id': worker_id.split('-')[-1],  # Short ID
                    'memory_limit_gb': info.get('memory_limit', 0) / (1024**3),
                    'memory_used_gb': info.get('memory', 0) / (1024**3),
                    'memory_usage_pct': (info.get('memory', 0) / info.get('memory_limit', 1)) * 100,
                    'status': info.get('status', 'unknown'),
                }
                
                memory_summary['workers_detail'].append(worker_memory)
                memory_summary['total_memory_limit'] += worker_memory['memory_limit_gb']
                memory_summary['total_memory_used'] += worker_memory['memory_used_gb']
            
            memory_summary['overall_usage_pct'] = (
                memory_summary['total_memory_used'] / memory_summary['total_memory_limit'] * 100 
                if memory_summary['total_memory_limit'] > 0 else 0
            )
            
            return memory_summary
        except Exception as e:
            self.logger.error(f"Error getting memory usage summary: {e}")
            return None
            
    def close(self):
        """Close the adaptive Dask client and cleanup resources"""
        # Stop monitoring first
        self.stop_adaptive_monitoring()
        
        if self.client:
            try:
                # Get final status before closing
                final_status = self.get_scaling_status()
                if final_status:
                    self.logger.info(f"Final adaptive status before close: {final_status}")
                
                # Close client with extended timeout for adaptive cleanup
                self.client.close(timeout=30)
                self.logger.info("Adaptive Dask client closed")
            except Exception as e:
                self.logger.warning(f"Error during client close (may be harmless): {e}")
                # Force close if graceful close fails
                try:
                    self.client.close(timeout=5)
                except:
                    pass
            finally:
                self.client = None
        
        # Close cluster
        if self.cluster:
            try:
                self.cluster.close(timeout=30)
                self.logger.info("Adaptive Dask cluster closed")
            except Exception as e:
                self.logger.warning(f"Error during cluster close (may be harmless): {e}")
            finally:
                self.cluster = None
                
    def __enter__(self):
        """Context manager entry - setup and create adaptive client"""
        self.setup_dask_config()
        self.create_client()
        self.wait_for_workers(min_workers=1)  # Only wait for minimum in adaptive mode
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleanup adaptive client"""
        self.close()
        # Suppress the timeout exception if it occurs during cleanup
        if exc_type is TimeoutError:
            self.logger.warning("Timeout during adaptive Dask cleanup - this is usually harmless")
            return True  # Suppress the exception
        return False


# Convenience function to maintain compatibility with existing code
def create_adaptive_dask_manager():
    """Factory function to create an AdaptiveDaskManager instance"""
    return AdaptiveDaskManager()