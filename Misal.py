class BatchedPairSampler:
    # ...

    def _sample_hard_negatives(self, domain: str, concept: str,
                              positive_attrs: pd.DataFrame, n_required: int,
                              attributes_df: pd.DataFrame) -> List[Tuple[str, str, float]]:
        """Sample hard negatives using batched operations."""
        hard_negatives = []
        other_concepts = [c for c in self.data_processor.get_concepts_for_domain(domain) 
                         if c != concept]
        
        self.logger.info(f"Sampling hard negatives for domain: {domain}, concept: {concept}")
        self.logger.info(f"Positive attributes shape: {positive_attrs.shape}")
        self.logger.info(f"Other concepts: {other_concepts}")
        
        if not other_concepts:
            self.logger.info("No other concepts found for the domain, skipping hard negatives.")
            return hard_negatives
            
        # Get potential negative attributes efficiently
        mask = (attributes_df['domain'] == domain) & (attributes_df['concept'].isin(other_concepts))
        potential_negatives = attributes_df[mask]
        
        if len(potential_negatives) == 0:
            self.logger.info("No potential negative attributes found, skipping hard negatives.")
            return hard_negatives
            
        # Prepare texts for batch processing
        pos_texts = [
            row['description'] for _, row in positive_attrs.iterrows()
        ]
        neg_texts = [
            row['description'] for _, row in potential_negatives.iterrows()
        ]
        
        self.logger.info(f"Positive texts count: {len(pos_texts)}")
        self.logger.info(f"Negative texts count: {len(neg_texts)}")
        
        # Compute similarities in batches
        batch_size = min(128, len(pos_texts))
        similarities = []
        
        for i in range(0, len(pos_texts), batch_size):
            pos_batch = pos_texts[i:i+batch_size]
            sim_batch = self.compute_batch_similarities(pos_batch, neg_texts)
            similarities.append(sim_batch)
            
        # Combine similarities
        similarities = torch.cat(similarities, dim=0)
        
        # Find high similarity pairs
        high_sim_indices = torch.nonzero(similarities > 0.8)
        
        self.logger.info(f"Number of high similarity pairs: {len(high_sim_indices)}")
        
        # Create hard negatives
        for pos_idx, neg_idx in high_sim_indices[:n_required]:
            neg_row = potential_negatives.iloc[neg_idx]
            neg_text = self.data_processor.get_attribute_text(
                neg_row['attribute_name'],
                neg_row['description']
            )
            neg_def = self.data_processor.get_concept_definition(domain, neg_row['concept'])
            hard_negatives.append((neg_text, neg_def, 1.5))
            
        return hard_negatives

    def _sample_medium_negatives(self, domain: str, concept: str,
                                n_required: int, attributes_df: pd.DataFrame) -> List[Tuple[str, str, float]]:
        """Sample medium negatives efficiently."""
        medium_negatives = []
        other_domains = [d for d in self.data_processor.get_domains() if d != domain]
        
        self.logger.info(f"Sampling medium negatives for domain: {domain}, concept: {concept}")
        self.logger.info(f"Other domains: {other_domains}")
        
        if not other_domains:
            self.logger.info("No other domains found, skipping medium negatives.")
            return medium_negatives
            
        # Batch sample from other domains
        sample_domains = np.random.choice(other_domains, n_required, replace=True)
        domain_batches = defaultdict(list)
        
        # Group by domain for efficient processing
        for d in sample_domains:
            domain_batches[d].append(d)
            
        # Process each domain batch
        for d, batch in domain_batches.items():
            concepts = self.data_processor.get_concepts_for_domain(d)
            if not concepts:
                self.logger.info(f"No concepts found for domain: {d}, skipping.")
                continue
                
            # Sample concepts for entire batch
            batch_concepts = np.random.choice(list(concepts), len(batch))
            
            # Get attributes efficiently
            mask = (attributes_df['domain'] == d) & (attributes_df['concept'].isin(set(batch_concepts)))
            neg_attrs = attributes_df[mask]
            
            if len(neg_attrs) == 0:
                self.logger.info(f"No negative attributes found for domain: {d}, skipping.")
                continue
                
            # Sample attributes for batch
            sampled_indices = np.random.randint(0, len(neg_attrs), len(batch))
            sampled_attrs = neg_attrs.iloc[sampled_indices]
            
            for _, row in sampled_attrs.iterrows():
                neg_text = self.data_processor.get_attribute_text(
                    row['attribute_name'],
                    row['description']
                )
                neg_def = self.data_processor.get_concept_definition(d, row['concept'])
                medium_negatives.append((neg_text, neg_def, 1.0))
                
                if len(medium_negatives) >= n_required:
                    return medium_negatives[:n_required]
                    
        return medium_negatives
