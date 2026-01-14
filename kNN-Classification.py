import numpy as np
import matplotlib.pyplot as plt
import random
import time

# Initial points for each class
INITIAL_POINTS = {
    'R': [[-4500, -4400], [-4100, -3000], [-1800, -2400], [-2500, -3400], [-2000, -1400]],
    'G': [[+4500, -4400], [+4100, -3000], [+1800, -2400], [+2500, -3400], [+2000, -1400]],
    'B': [[-4500, +4400], [-4100, +3000], [-1800, +2400], [-2500, +3400], [-2000, +1400]],
    'P': [[+4500, +4400], [+4100, +3000], [+1800, +2400], [+2500, +3400], [+2000, +1400]]
}

CLASS_COLORS = {'R': 'red', 'G': 'green', 'B': 'blue', 'P': 'purple'}


class KNNClassifier:
    def __init__(self):
        self.points = []  # List of [x, y, class]
        self.coords_array = None
        self.classes_array = None
        self.reset()

    def reset(self):
        self.points = []
        for class_name, coords_list in INITIAL_POINTS.items():
            for coords in coords_list:
                self.points.append([coords[0], coords[1], class_name])
        self._update_arrays()

    def _update_arrays(self):
        if len(self.points) > 0:
            # Extract coordinates and classes
            coords = [[p[0], p[1]] for p in self.points]
            classes = [p[2] for p in self.points]

            self.coords_array = np.array(coords, dtype=np.float32)
            self.classes_array = np.array(classes)

    def find_k_nearest(self, x, y, k):
        query = np.array([x, y], dtype=np.float32)

        distances = np.sqrt(np.sum((self.coords_array - query) ** 2, axis=1))

        k_nearest_indices = np.argpartition(distances, min(k, len(distances) - 1))[:k]
        k_nearest_classes = self.classes_array[k_nearest_indices]

        return k_nearest_classes.tolist()

    def classify(self, x, y, k):
        # Find k nearest neighbors
        neighbors = self.find_k_nearest(x, y, k)

        # Vote: most common class wins
        vote_counts = {}
        for neighbor_class in neighbors:
            if neighbor_class in vote_counts:
                vote_counts[neighbor_class] += 1
            else:
                vote_counts[neighbor_class] = 1

        predicted_class = max(vote_counts, key=vote_counts.get)

        # Add classified point to our space
        self.points.append([x, y, predicted_class])

        # Update arrays every 100 points for efficiency
        if len(self.points) % 100 == 0:
            self._update_arrays()

        return predicted_class

    def get_points_by_class(self):
        by_class = {'R': [], 'G': [], 'B': [], 'P': []}
        for point in self.points:
            by_class[point[2]].append([point[0], point[1]])
        return by_class


def generate_test_points(num_per_class=10000):
    test_points = []
    classes = ['R', 'G', 'B', 'P']

    print(f"Generating {num_per_class * 4} test points")

    for i in range(num_per_class * 4):
        class_idx = i % 4
        true_class = classes[class_idx]

        # 99% probability in preferred region, 1% anywhere
        if random.random() < 0.99:
            if true_class == 'R':
                x = random.randint(-5000, 500)
                y = random.randint(-5000, 500)
            elif true_class == 'G':
                x = random.randint(-500, 5000)
                y = random.randint(-5000, 500)
            elif true_class == 'B':
                x = random.randint(-5000, 500)
                y = random.randint(-500, 5000)
            else:  # P
                x = random.randint(-500, 5000)
                y = random.randint(-500, 5000)
        else:
            x = random.randint(-5000, 5000)
            y = random.randint(-5000, 5000)

        test_points.append([x, y, true_class])

    return test_points


def run_experiment(k_value, test_points):
    print(f"Running experiment with k={k_value}")

    classifier = KNNClassifier()
    correct = 0
    total = len(test_points)

    start_time = time.time()

    for i, (x, y, true_class) in enumerate(test_points):
        predicted_class = classifier.classify(x, y, k_value)

        if predicted_class == true_class:
            correct += 1

        # Progress indicator
        if (i + 1) % 1000 == 0:
            progress = (i + 1) / total * 100

            print(f"Progress: {i + 1}/{total} | "
                  f"Accuracy: {correct / (i + 1) * 100:.2f}% | ")


    # Final array update
    classifier._update_arrays()

    accuracy = (correct / total) * 100
    elapsed_time = time.time() - start_time

    print(f"\nExperiment completed!")
    print(f"k={k_value}: Accuracy = {accuracy:.2f}")
    print(f"Time taken: {elapsed_time:.1f} seconds")

    return classifier, accuracy


def visualize_results(classifier, k_value, accuracy):
    fig, ax = plt.subplots(figsize=(12, 12))

    points_by_class = classifier.get_points_by_class()

    for class_name, color in CLASS_COLORS.items():
        points = np.array(points_by_class[class_name])
        if len(points) > 0:
            ax.scatter(points[:, 0], points[:, 1],
                       c=color, s=100, alpha=0.6, label=class_name)

    ax.set_xlim([-5000, 5000])
    ax.set_ylim([-5000, 5000])
    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('Y', fontsize=12)
    ax.set_title(f'k-NN Classification (k={k_value}) - Accuracy: {accuracy:.2f}%',
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.3, linewidth=0.5)
    ax.axvline(x=0, color='k', linestyle='-', alpha=0.3, linewidth=0.5)

    plt.tight_layout()
    plt.savefig(f'knn_k{k_value}.png', dpi=150)
    plt.show()

def main():
    print("Starting k-NN Experiments")


    # Generate test points once (same for all experiments)
    test_points = generate_test_points(num_per_class=10000)

    # Run experiments for different k values
    k_values = [1, 3, 7, 15]
    results = {}

    for k in k_values:
        classifier, accuracy = run_experiment(k, test_points)
        results[k] = accuracy
        visualize_results(classifier, k, accuracy)



    # Compare k values
    best_k = max(results, key=results.get)
    print(f"\nBest performing k: {best_k} with {results[best_k]:.2f}% accuracy")


if __name__ == "__main__":
    main()