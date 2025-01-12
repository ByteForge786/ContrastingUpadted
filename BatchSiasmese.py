def _save_model(self, model: nn.Module, name: str) -> str:
        """Save model and return path."""
        save_path = self.output_dir / name
        os.makedirs(save_path, exist_ok=True)
        
        # Save encoder
        model.encoder.save(save_path)
        
        # Save full model state
        model_path = save_path / 'model_state.pt'
        torch.save(model.state_dict(), model_path)
        
        # Save config
        config_path = save_path / 'config.json'
        with open(config_path, 'w') as f:
            json.dump(vars(self.config), f, indent=2)
            
        return str(save_path)
    
    def _save_training_results(self, fold_results: List[Dict], test_loss: float, test_acc: float) -> None:
        """Save training results and metrics."""
        results = {
            'fold_results': fold_results,
            'test_metrics': {
                'loss': test_loss,
                'accuracy': test_acc
            },
            'training_metrics': dict(self.metrics),
            'config': vars(self.config),
            'best_model_path': self.best_model_path,
            'timestamp': datetime.now().isoformat()
        }
        
        results_path = self.output_dir / 'training_results.json'
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        self.logger.info(f"Saved training results to {results_path}")

class PredictionService:
    def __init__(self, model_path: str, config: Optional[TrainingConfig] = None):
        self.logger = logging.getLogger(__name__)
        self.model_path = Path(model_path)
        
        # Load config
        config_path = self.model_path / 'config.json'
        if config is None:
            with open(config_path) as f:
                config_dict = json.load(f)
                self.config = TrainingConfig(**config_dict)
        else:
            self.config = config
            
        # Initialize model
        self.model = self._load_model()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()
        
    def _load_model(self) -> EnhancedSiameseNetwork:
        """Load the trained model."""
        try:
            model = EnhancedSiameseNetwork(
                self.config.model_name,
                self.config.hidden_size,
                self.config.dropout_rate
            )
            
            state_dict_path = self.model_path / 'model_state.pt'
            state_dict = torch.load(state_dict_path)
            model.load_state_dict(state_dict)
            
            return model
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            raise
            
    @torch.no_grad()
    def predict_single(self, attribute_name: str, description: str, 
                      concept_texts: List[str], top_k: int = 3) -> List[Dict]:
        """Predict for a single attribute."""
        self.model.eval()
        
        # Prepare input
        attribute_text = f"{attribute_name} {description}"
        attribute_texts = [attribute_text] * len(concept_texts)
        
        # Get predictions
        outputs = self.model(attribute_texts, concept_texts)
        probabilities = torch.sigmoid(outputs)
        
        # Get top-k predictions
        top_k_values, top_k_indices = torch.topk(probabilities, min(top_k, len(concept_texts)))
        
        # Prepare results
        results = []
        for i, (prob, idx) in enumerate(zip(top_k_values, top_k_indices)):
            concept_text = concept_texts[idx]
            domain, concept = concept_text.split(':')[0].strip().split('-')
            
            results.append({
                'rank': i + 1,
                'domain': domain,
                'concept': concept,
                'confidence': float(prob)
            })
            
        return results
    
    def predict_batch(self, df: pd.DataFrame, concept_texts: List[str], 
                     output_path: Optional[str] = None) -> pd.DataFrame:
        """Predict for a batch of attributes."""
        self.logger.info(f"Processing {len(df)} attributes...")
        
        results = []
        with ThreadPoolExecutor(max_workers=self.config.num_workers) as executor:
            futures = []
            
            for _, row in df.iterrows():
                futures.append(
                    executor.submit(
                        self.predict_single,
                        row['attribute_name'],
                        row['description'],
                        concept_texts
                    )
                )
            
            for future in tqdm(futures, desc="Processing predictions"):
                predictions = future.result()
                if predictions:
                    top_pred = predictions[0]  # Get top prediction
                    results.append({
                        'predicted_domain': top_pred['domain'],
                        'predicted_concept': top_pred['concept'],
                        'confidence': top_pred['confidence']
                    })
                else:
                    results.append({
                        'predicted_domain': None,
                        'predicted_concept': None,
                        'confidence': 0.0
                    })
        
        # Create results DataFrame
        results_df = pd.concat([
            df,
            pd.DataFrame(results)
        ], axis=1)
        
        # Save if output path provided
        if output_path:
            results_df.to_csv(output_path, index=False)
            self.logger.info(f"Saved predictions to {output_path}")
            
        return results_df

def main():
    parser = argparse.ArgumentParser(description='Enhanced Siamese Network for Text Matching')
    
    # Common arguments
    parser.add_argument('--mode', choices=['train', 'predict'], required=True,
                      help='Operation mode')
    parser.add_argument('--attributes', type=str, required=True,
                      help='Path to attributes CSV')
    parser.add_argument('--concepts', type=str, required=True,
                      help='Path to concepts CSV')
    parser.add_argument('--output-dir', type=str, default='./output',
                      help='Output directory')
    
    # Training arguments
    parser.add_argument('--batch-size', type=int, default=32,
                      help='Batch size')
    parser.add_argument('--epochs', type=int, default=50,
                      help='Number of epochs')
    parser.add_argument('--learning-rate', type=float, default=2e-5,
                      help='Learning rate')
    parser.add_argument('--num-folds', type=int, default=5,
                      help='Number of CV folds')
    
    # Prediction arguments
    parser.add_argument('--model-path', type=str,
                      help='Path to trained model')
    parser.add_argument('--input', type=str,
                      help='Input file for prediction')
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logger(args.output_dir)
    
    try:
        if args.mode == 'train':
            # Create config
            config = TrainingConfig(
                batch_size=args.batch_size,
                num_epochs=args.epochs,
                learning_rate=args.learning_rate,
                num_folds=args.num_folds
            )
            
            # Initialize components
            data_processor = DataProcessor(args.attributes, args.concepts, config)
            train_dataset, test_dataset = data_processor.process_data()
            
            trainer = ModelTrainer(config, args.output_dir)
            trainer.train(train_dataset, test_dataset)
            
        else:  # predict mode
            if not args.model_path or not args.input:
                raise ValueError("--model-path and --input required for predict mode")
                
            # Load concepts for prediction
            concepts_df = pd.read_csv(args.concepts)
            concept_texts = [
                f"{row['domain']}-{row['concept']}: {row['concept_definition']}"
                for _, row in concepts_df.iterrows()
            ]
            
            # Initialize prediction service
            predictor = PredictionService(args.model_path)
            
            # Read input and make predictions
            input_df = pd.read_csv(args.input)
            output_path = os.path.join(args.output_dir, 'predictions.csv')
            predictor.predict_batch(input_df, concept_texts, output_path)
            
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        raise

if __name__ == "__main__":
    main()
