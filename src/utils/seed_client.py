# Copyright 2025 POLAR Team and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
SEED Score Client

Client library for communicating with the SEED Score Server.
Use this in your GPU training programs to compute SEED scores remotely.

Example:
    from seed_client import SEEDClient
    
    client = SEEDClient(server_address="127.0.0.1:30001")
    
    # Single computation
    result = client.compute(reference="x^2 + 1", output="x^2+1")
    print(f"Score: {result['score']}")
    
    # Batch computation
    results = client.compute_batch(
        references=["x^2", "\\frac{1}{2}"],
        outputs=["x^2", "0.5"]
    )
    print(f"Scores: {results['scores']}")
"""

import requests
from typing import List, Dict, Any, Optional, Union
from time import sleep
import logging

logger = logging.getLogger('SEEDClient')


class SEEDClient:
    """
    Client for SEED Score Computation Server.
    
    Provides methods to compute SEED scores via HTTP requests to the server.
    Communication pattern follows src/polar/reward_func.py style.
    """
    
    def __init__(
        self,
        server_address: str = "127.0.0.1:30001",
        timeout: float = 120.0,
        max_retries: int = 3,
        retry_delay: float = 0.5
    ):
        """
        Initialize the SEED client.
        
        Args:
            server_address: Address of the SEED server (host:port).
            timeout: Request timeout in seconds.
            max_retries: Maximum number of retry attempts.
            retry_delay: Delay between retries in seconds.
        """
        self.server_address = server_address
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        # Ensure no http:// prefix stored
        if self.server_address.startswith('http://'):
            self.server_address = self.server_address[7:]
        elif self.server_address.startswith('https://'):
            self.server_address = self.server_address[8:]
    
    def _make_request(
        self,
        endpoint: str,
        data: Dict[str, Any],
        method: str = 'POST'
    ) -> Optional[Dict[str, Any]]:
        """
        Make an HTTP request to the server with retry logic.
        
        Args:
            endpoint: API endpoint (e.g., '/compute').
            data: Request payload.
            method: HTTP method.
            
        Returns:
            Response JSON or None if all retries failed.
        """
        url = f"http://{self.server_address}{endpoint}"
        
        for attempt in range(self.max_retries):
            try:
                if method == 'POST':
                    response = requests.post(
                        url,
                        json=data,
                        proxies={"http": None, "https": None},  # Disable proxy
                        timeout=self.timeout
                    )
                else:
                    response = requests.get(
                        url,
                        proxies={"http": None, "https": None},
                        timeout=self.timeout
                    )
                
                if response.status_code == 200:
                    return response.json()
                else:
                    logger.warning(
                        f"Request failed with status {response.status_code}: {response.text}"
                    )
                    
            except requests.exceptions.Timeout:
                logger.warning(f"Request timeout (attempt {attempt + 1}/{self.max_retries})")
            except requests.exceptions.ConnectionError as e:
                logger.warning(f"Connection error (attempt {attempt + 1}/{self.max_retries}): {e}")
            except Exception as e:
                logger.error(f"Unexpected error: {e}")
            
            if attempt < self.max_retries - 1:
                sleep(self.retry_delay * (attempt + 1))  # Exponential backoff
        
        logger.error(f"All {self.max_retries} attempts failed for {endpoint}")
        return None
    
    def health_check(self) -> bool:
        """
        Check if the server is healthy.
        
        Returns:
            True if server is healthy, False otherwise.
        """
        try:
            response = requests.get(
                f"http://{self.server_address}/health",
                proxies={"http": None, "https": None},
                timeout=5.0
            )
            return response.status_code == 200
        except Exception:
            return False
    
    def get_stats(self) -> Optional[Dict[str, Any]]:
        """
        Get server statistics.
        
        Returns:
            Dictionary with server stats or None if failed.
        """
        return self._make_request('/stats', {}, method='GET')
    
    def compute(
        self,
        reference: str,
        output: str,
        expr_type: str = "Expression"
    ) -> Optional[Dict[str, Any]]:
        """
        Compute SEED score for a single pair.
        
        Args:
            reference: Ground truth LaTeX expression.
            output: Model output LaTeX expression.
            expr_type: Expression type (Expression, Equation, Tuple, Interval, Numeric).
            
        Returns:
            Dictionary with score, relative_distance, tree_size, distance, error.
            Returns None if request failed.
        """
        data = {
            "reference": reference,
            "output": output,
            "type": expr_type
        }
        return self._make_request('/compute', data)
    
    def compute_batch(
        self,
        references: List[str],
        outputs: List[str],
        types: Optional[List[str]] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Compute SEED scores for a batch of pairs.
        
        Args:
            references: List of ground truth LaTeX expressions.
            outputs: List of model output LaTeX expressions.
            types: List of expression types (default: all "Expression").
            
        Returns:
            Dictionary with 'results' (detailed) and 'scores' (list of float).
            Returns None if request failed.
        """
        if types is None:
            types = ["Expression"] * len(references)
        
        data = {
            "references": references,
            "outputs": outputs,
            "types": types
        }
        return self._make_request('/compute_batch', data)
    
    def compute_batch_items(
        self,
        items: List[Dict[str, str]]
    ) -> Optional[Dict[str, Any]]:
        """
        Compute SEED scores using item format.
        
        Args:
            items: List of dicts with 'reference', 'output', and optional 'type'.
            
        Returns:
            Dictionary with 'results' and 'scores'.
        """
        data = {"items": items}
        return self._make_request('/compute_batch', data)
    
    def __call__(
        self,
        references: Union[str, List[str]],
        outputs: Union[str, List[str]],
        types: Optional[Union[str, List[str]]] = None
    ) -> Optional[Union[float, List[float]]]:
        """
        Convenience method to compute scores.
        
        Args:
            references: Single reference or list of references.
            outputs: Single output or list of outputs.
            types: Single type or list of types.
            
        Returns:
            Single score (float) or list of scores.
            Returns 0.0 for failed single requests, list of 0.0s for failed batch.
        """
        is_single = isinstance(references, str)
        
        if is_single:
            result = self.compute(
                reference=references,
                output=outputs,
                expr_type=types if types else "Expression"
            )
            if result is None:
                return 0.0
            return result.get('score', 0.0)
        else:
            result = self.compute_batch(
                references=references,
                outputs=outputs,
                types=types
            )
            if result is None:
                return [0.0] * len(references)
            return result.get('scores', [0.0] * len(references))


# Global client instance
_seed_client: Optional[SEEDClient] = None


def get_seed_client(
    server_address: str = "127.0.0.1:30001",
    timeout: float = 120.0
) -> SEEDClient:
    """
    Get or create a global SEED client instance.
    
    Args:
        server_address: SEED server address.
        timeout: Request timeout.
        
    Returns:
        SEEDClient instance.
    """
    global _seed_client
    if _seed_client is None:
        _seed_client = SEEDClient(
            server_address=server_address,
            timeout=timeout
        )
    return _seed_client


# Convenience function for direct usage
def compute_seed_scores(
    references: List[str],
    outputs: List[str],
    types: Optional[List[str]] = None,
    server_address: str = "127.0.0.1:30001"
) -> List[float]:
    """
    Compute SEED scores using the server.
    
    This is a convenience function for direct usage in reward computation.
    
    Args:
        references: List of ground truth expressions.
        outputs: List of model outputs.
        types: List of expression types.
        server_address: SEED server address.
        
    Returns:
        List of SEED scores (0.0 for failures).
    """
    client = get_seed_client(server_address=server_address)
    scores = client(references, outputs, types)
    return scores if isinstance(scores, list) else [scores]


if __name__ == '__main__':
    # Test the client
    import argparse
    
    parser = argparse.ArgumentParser(description='Test SEED Client')
    parser.add_argument('--server', type=str, default='10.102.241.25:30030',
                        help='SEED server address')
    args = parser.parse_args()
    
    client = SEEDClient(server_address=args.server)
    
    # Health check
    print(f"Server healthy: {client.health_check()}")
    
    # Single computation
    print("\n--- Single Computation ---")
    result = client.compute(
        reference="x^2 + 2x + 1",
        output="(x+1)^2"
    )
    print(f"Result: {result}")
    
    # Batch computation
    print("\n--- Batch Computation ---")
    result = client.compute_batch(
        references=["x^2", "\\frac{1}{2}", "cos(x)"],
        outputs=["x^2", "0.5", "sin(x)"]
    )
    print(f"Scores: {result['scores'] if result else 'Failed'}")
    
    # Using __call__
    print("\n--- Using __call__ ---")
    scores = client(
        references=[r"[0,1]", "2x"],
        outputs=[r"\left[0, 1 \right]", "x + x"]
    )
    print(f"Scores: {scores}")
