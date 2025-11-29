
from metrics_evaluator import evaluate_metrics
from graph_generator import generate_graphs
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    """
    Main function to run the evaluation pipeline.
    """
    logging.info("Starting the evaluation pipeline...")

    # In a real-world scenario, you would load your generated and reference texts here.
    # For demonstration purposes, we'll use the sample data from the evaluator.
    sample_references = [
        "The cat sat on the mat.",
        "The quick brown fox jumps over the lazy dog.",
        "She sells sea shells by the sea shore."
    ]
    sample_candidates = [
        "A cat is on the mat.",
        "A fast brown fox leaps over a lazy dog.",
        "She sells sea-shells on the sea-shore."
    ]

    logging.info("Step 1: Calculating and logging metrics...")
    evaluate_metrics(sample_references, sample_candidates)
    logging.info("Metrics calculation and logging complete.")

    logging.info("Step 2: Generating graphs...")
    generate_graphs()
    logging.info("Graph generation complete.")

    logging.info("Evaluation pipeline finished successfully.")

if __name__ == '__main__':
    main()
