#!/usr/bin/env python3
"""
Experiment Tracking Database for VariBAD
Phase 2: Systematic experiment management and comparison
"""

import sqlite3
import json
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import numpy as np

class ExperimentDatabase:
    """Database for tracking VariBAD experiments and results"""
    
    def __init__(self, db_path: str = "results/experiments.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize_database()
    
    def _initialize_database(self):
        """Initialize database tables"""
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Experiments table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS experiments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT UNIQUE NOT NULL,
                    description TEXT,
                    status TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    started_at TIMESTAMP,
                    completed_at TIMESTAMP,
                    config_json TEXT,
                    checkpoint_path TEXT,
                    results_path TEXT,
                    tags TEXT,
                    final_reward REAL,
                    avg_reward REAL,
                    best_reward REAL,
                    total_iterations INTEGER,
                    training_time_seconds REAL,
                    model_parameters INTEGER
                )
            """)
            
            # Hyperparameters table (for easy querying)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS hyperparameters (
                    experiment_id INTEGER,
                    param_name TEXT NOT NULL,
                    param_value TEXT NOT NULL,
                    param_type TEXT,
                    FOREIGN KEY (experiment_id) REFERENCES experiments (id),
                    PRIMARY KEY (experiment_id, param_name)
                )
            """)
            
            # Training metrics table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS training_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    experiment_id INTEGER,
                    iteration INTEGER,
                    episode_reward REAL,
                    vae_loss REAL,
                    policy_loss REAL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (experiment_id) REFERENCES experiments (id)
                )
            """)
            
            # Comparisons table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS comparisons (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    description TEXT,
                    experiment_ids TEXT,
                    comparison_type TEXT,
                    results_json TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.commit()
    
    def register_experiment(self, config: Dict[str, Any], experiment_name: str) -> int:
        """Register a new experiment and return its ID"""
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Extract key information
            experiment_info = config.get('experiment', {})
            training_info = config.get('training', {})
            
            cursor.execute("""
                INSERT INTO experiments (
                    name, description, status, config_json, tags,
                    total_iterations
                ) VALUES (?, ?, ?, ?, ?, ?)
            """, (
                experiment_name,
                experiment_info.get('description', ''),
                'registered',
                json.dumps(config),
                ','.join(experiment_info.get('tags', [])),
                training_info.get('num_iterations', 0)
            ))
            
            experiment_id = cursor.lastrowid
            
            # Store hyperparameters for easy querying
            self._store_hyperparameters(cursor, experiment_id, config)
            
            conn.commit()
            return experiment_id
    
    def _store_hyperparameters(self, cursor, experiment_id: int, config: Dict[str, Any]):
        """Store flattened hyperparameters for easy querying"""
        
        def flatten_config(obj, prefix=""):
            """Flatten nested config dictionary"""
            params = {}
            for key, value in obj.items():
                if key.startswith('_'):  # Skip metadata
                    continue
                    
                full_key = f"{prefix}.{key}" if prefix else key
                
                if isinstance(value, dict):
                    params.update(flatten_config(value, full_key))
                else:
                    params[full_key] = value
            return params
        
        params = flatten_config(config)
        
        for param_name, param_value in params.items():
            param_type = type(param_value).__name__
            cursor.execute("""
                INSERT OR REPLACE INTO hyperparameters 
                (experiment_id, param_name, param_value, param_type)
                VALUES (?, ?, ?, ?)
            """, (experiment_id, param_name, str(param_value), param_type))
    
    def start_experiment(self, experiment_id: int):
        """Mark experiment as started"""
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE experiments 
                SET status = 'running', started_at = CURRENT_TIMESTAMP
                WHERE id = ?
            """, (experiment_id,))
            conn.commit()
    
    def complete_experiment(self, experiment_id: int, 
                          checkpoint_path: str = None,
                          results_path: str = None,
                          final_reward: float = None,
                          avg_reward: float = None,
                          best_reward: float = None,
                          training_time: float = None,
                          model_parameters: int = None):
        """Mark experiment as completed with results"""
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE experiments 
                SET status = 'completed', completed_at = CURRENT_TIMESTAMP,
                    checkpoint_path = ?, results_path = ?, final_reward = ?,
                    avg_reward = ?, best_reward = ?, training_time_seconds = ?,
                    model_parameters = ?
                WHERE id = ?
            """, (checkpoint_path, results_path, final_reward, avg_reward, 
                  best_reward, training_time, model_parameters, experiment_id))
            conn.commit()
    
    def fail_experiment(self, experiment_id: int, error_message: str = None):
        """Mark experiment as failed"""
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE experiments 
                SET status = 'failed', completed_at = CURRENT_TIMESTAMP,
                    description = COALESCE(description, '') || ' ERROR: ' || ?
                WHERE id = ?
            """, (error_message or 'Unknown error', experiment_id))
            conn.commit()
    
    def log_training_metrics(self, experiment_id: int, iteration: int,
                           episode_reward: float = None,
                           vae_loss: float = None,
                           policy_loss: float = None):
        """Log training metrics for an iteration"""
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO training_metrics 
                (experiment_id, iteration, episode_reward, vae_loss, policy_loss)
                VALUES (?, ?, ?, ?, ?)
            """, (experiment_id, iteration, episode_reward, vae_loss, policy_loss))
            conn.commit()
    
    def get_experiments(self, status: str = None, 
                       tags: List[str] = None,
                       limit: int = None) -> pd.DataFrame:
        """Query experiments with filters"""
        
        query = "SELECT * FROM experiments"
        params = []
        
        conditions = []
        if status:
            conditions.append("status = ?")
            params.append(status)
        
        if tags:
            for tag in tags:
                conditions.append("tags LIKE ?")
                params.append(f"%{tag}%")
        
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        
        query += " ORDER BY created_at DESC"
        
        if limit:
            query += f" LIMIT {limit}"
        
        with sqlite3.connect(self.db_path) as conn:
            return pd.read_sql_query(query, conn, params=params)
    
    def get_hyperparameter_comparison(self, experiment_ids: List[int]) -> pd.DataFrame:
        """Get hyperparameter comparison for experiments"""
        
        placeholders = ','.join(['?'] * len(experiment_ids))
        query = f"""
            SELECT e.name, h.param_name, h.param_value
            FROM experiments e
            JOIN hyperparameters h ON e.id = h.experiment_id
            WHERE e.id IN ({placeholders})
            ORDER BY h.param_name, e.name
        """
        
        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql_query(query, conn, params=experiment_ids)
            
            # Pivot to get experiments as columns
            return df.pivot(index='param_name', columns='name', values='param_value')
    
    def get_training_curves(self, experiment_ids: List[int]) -> pd.DataFrame:
        """Get training curves for experiments"""
        
        placeholders = ','.join(['?'] * len(experiment_ids))
        query = f"""
            SELECT e.name, m.iteration, m.episode_reward, m.vae_loss, m.policy_loss
            FROM experiments e
            JOIN training_metrics m ON e.id = m.experiment_id
            WHERE e.id IN ({placeholders})
            ORDER BY e.name, m.iteration
        """
        
        with sqlite3.connect(self.db_path) as conn:
            return pd.read_sql_query(query, conn, params=experiment_ids)
    
    def find_best_experiments(self, metric: str = 'final_reward', 
                             limit: int = 10) -> pd.DataFrame:
        """Find best performing experiments"""
        
        query = f"""
            SELECT name, description, {metric}, status, tags, created_at
            FROM experiments
            WHERE status = 'completed' AND {metric} IS NOT NULL
            ORDER BY {metric} DESC
            LIMIT ?
        """
        
        with sqlite3.connect(self.db_path) as conn:
            return pd.read_sql_query(query, conn, params=[limit])
    
    def create_comparison(self, name: str, experiment_ids: List[int],
                         comparison_type: str = 'performance',
                         description: str = None) -> int:
        """Create a saved comparison between experiments"""
        
        # Get experiment data
        experiments_df = self.get_experiments()
        selected_experiments = experiments_df[experiments_df['id'].isin(experiment_ids)]
        
        # Get hyperparameter comparison
        hyperparams_df = self.get_hyperparameter_comparison(experiment_ids)
        
        # Get training curves
        training_df = self.get_training_curves(experiment_ids)
        
        # Create comparison results
        comparison_results = {
            'experiments': selected_experiments.to_dict('records'),
            'hyperparameters': hyperparams_df.to_dict(),
            'training_curves': training_df.to_dict('records'),
            'summary': {
                'total_experiments': len(experiment_ids),
                'best_final_reward': float(selected_experiments['final_reward'].max()) if not selected_experiments['final_reward'].isna().all() else None,
                'avg_final_reward': float(selected_experiments['final_reward'].mean()) if not selected_experiments['final_reward'].isna().all() else None
            }
        }
        
        # Store comparison
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO comparisons (name, description, experiment_ids, comparison_type, results_json)
                VALUES (?, ?, ?, ?, ?)
            """, (name, description, ','.join(map(str, experiment_ids)), 
                  comparison_type, json.dumps(comparison_results, default=str)))
            
            comparison_id = cursor.lastrowid
            conn.commit()
            
        return comparison_id
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """Get database summary statistics"""
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Basic counts
            cursor.execute("SELECT COUNT(*) FROM experiments")
            total_experiments = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM experiments WHERE status = 'completed'")
            completed_experiments = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM experiments WHERE status = 'running'")
            running_experiments = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM experiments WHERE status = 'failed'")
            failed_experiments = cursor.fetchone()[0]
            
            # Performance stats
            cursor.execute("SELECT MAX(final_reward), MIN(final_reward), AVG(final_reward) FROM experiments WHERE status = 'completed'")
            reward_stats = cursor.fetchone()
            
            return {
                'total_experiments': total_experiments,
                'completed': completed_experiments,
                'running': running_experiments,
                'failed': failed_experiments,
                'success_rate': completed_experiments / total_experiments if total_experiments > 0 else 0,
                'best_reward': reward_stats[0],
                'worst_reward': reward_stats[1],
                'avg_reward': reward_stats[2]
            }


def create_experiment_cli():
    """Command-line interface for experiment database management"""
    
    import argparse
    
    parser = argparse.ArgumentParser(description="VariBAD Experiment Database Management")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # List experiments
    list_parser = subparsers.add_parser('list', help='List experiments')
    list_parser.add_argument('--status', choices=['registered', 'running', 'completed', 'failed'])
    list_parser.add_argument('--tags', nargs='+', help='Filter by tags')
    list_parser.add_argument('--limit', type=int, default=20, help='Limit results')
    
    # Show experiment
    show_parser = subparsers.add_parser('show', help='Show experiment details')
    show_parser.add_argument('experiment_id', type=int, help='Experiment ID')
    
    # Compare experiments
    compare_parser = subparsers.add_parser('compare', help='Compare experiments')
    compare_parser.add_argument('experiment_ids', nargs='+', type=int, help='Experiment IDs to compare')
    compare_parser.add_argument('--name', required=True, help='Comparison name')
    compare_parser.add_argument('--description', help='Comparison description')
    
    # Best experiments
    best_parser = subparsers.add_parser('best', help='Show best experiments')
    best_parser.add_argument('--metric', default='final_reward', help='Metric to rank by')
    best_parser.add_argument('--limit', type=int, default=10, help='Number of results')
    
    # Summary
    summary_parser = subparsers.add_parser('summary', help='Database summary')
    
    args = parser.parse_args()
    
    # Initialize database
    db = ExperimentDatabase()
    
    if args.command == 'list':
        experiments = db.get_experiments(
            status=args.status,
            tags=args.tags,
            limit=args.limit
        )
        
        if len(experiments) == 0:
            print("No experiments found")
        else:
            # Display key columns
            display_cols = ['id', 'name', 'status', 'final_reward', 'created_at']
            available_cols = [col for col in display_cols if col in experiments.columns]
            print(experiments[available_cols].to_string(index=False))
    
    elif args.command == 'show':
        with sqlite3.connect(db.db_path) as conn:
            # Get experiment details
            exp_df = pd.read_sql_query(
                "SELECT * FROM experiments WHERE id = ?", 
                conn, params=[args.experiment_id]
            )
            
            if len(exp_df) == 0:
                print(f"Experiment {args.experiment_id} not found")
                return
            
            exp = exp_df.iloc[0]
            
            print(f"Experiment: {exp['name']}")
            print(f"Status: {exp['status']}")
            print(f"Description: {exp['description']}")
            print(f"Created: {exp['created_at']}")
            if exp['final_reward']:
                print(f"Final Reward: {exp['final_reward']:.4f}")
            
            # Get hyperparameters
            params_df = pd.read_sql_query(
                "SELECT param_name, param_value FROM hyperparameters WHERE experiment_id = ?",
                conn, params=[args.experiment_id]
            )
            
            if len(params_df) > 0:
                print("\nHyperparameters:")
                for _, row in params_df.iterrows():
                    print(f"  {row['param_name']}: {row['param_value']}")
    
    elif args.command == 'compare':
        comparison_id = db.create_comparison(
            name=args.name,
            experiment_ids=args.experiment_ids,
            description=args.description
        )
        
        print(f"Created comparison '{args.name}' with ID {comparison_id}")
        
        # Show basic comparison
        experiments = db.get_experiments()
        selected = experiments[experiments['id'].isin(args.experiment_ids)]
        
        print("\nExperiments in comparison:")
        display_cols = ['id', 'name', 'final_reward', 'status']
        available_cols = [col for col in display_cols if col in selected.columns]
        print(selected[available_cols].to_string(index=False))
    
    elif args.command == 'best':
        best_experiments = db.find_best_experiments(
            metric=args.metric,
            limit=args.limit
        )
        
        if len(best_experiments) == 0:
            print("No completed experiments found")
        else:
            print(f"Top {args.limit} experiments by {args.metric}:")
            print(best_experiments.to_string(index=False))
    
    elif args.command == 'summary':
        stats = db.get_summary_stats()
        
        print("Database Summary:")
        print(f"  Total experiments: {stats['total_experiments']}")
        print(f"  Completed: {stats['completed']}")
        print(f"  Running: {stats['running']}")
        print(f"  Failed: {stats['failed']}")
        print(f"  Success rate: {stats['success_rate']:.1%}")
        
        if stats['best_reward']:
            print(f"  Best reward: {stats['best_reward']:.4f}")
            print(f"  Average reward: {stats['avg_reward']:.4f}")
    
    else:
        parser.print_help()


def test_experiment_database():
    """Test the experiment database functionality"""
    
    print("🧪 Testing Experiment Database")
    print("=" * 40)
    
    # Create test database
    test_db = ExperimentDatabase("test_experiments.db")
    
    # Test 1: Register experiment
    print("1. Registering test experiment...")
    test_config = {
        'experiment': {
            'name': 'test_experiment',
            'description': 'Test experiment for database',
            'tags': ['test', 'database']
        },
        'training': {
            'num_iterations': 100,
            'episode_length': 30
        },
        'varibad': {
            'latent_dim': 5
        }
    }
    
    exp_id = test_db.register_experiment(test_config, 'test_experiment_1')
    print(f"✓ Registered experiment with ID: {exp_id}")
    
    # Test 2: Start and complete experiment
    print("2. Simulating experiment lifecycle...")
    test_db.start_experiment(exp_id)
    
    # Log some training metrics
    for i in range(5):
        test_db.log_training_metrics(
            exp_id, i, 
            episode_reward=0.1 + i * 0.01,
            vae_loss=1.0 - i * 0.1
        )
    
    test_db.complete_experiment(
        exp_id,
        checkpoint_path="test_checkpoint.pt",
        final_reward=0.15,
        avg_reward=0.13,
        training_time=300.0
    )
    print("✓ Completed experiment lifecycle")
    
    # Test 3: Query experiments
    print("3. Testing queries...")
    experiments = test_db.get_experiments(status='completed')
    print(f"✓ Found {len(experiments)} completed experiments")
    
    # Test 4: Get summary
    print("4. Getting summary...")
    stats = test_db.get_summary_stats()
    print(f"✓ Database contains {stats['total_experiments']} experiments")
    
    # Cleanup
    import os
    os.remove("test_experiments.db")
    print("✓ Test database cleaned up")
    
    print("\n🎉 All database tests passed!")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        create_experiment_cli()
    else:
        test_experiment_database()