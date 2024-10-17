import numpy as np
from scipy import stats
import math

buildings_data = {
    "Building_1": {
        "gross_area": 800,
        "type": "MFH"
    },
    "Building_2": {
        "gross_area": 600,
        "type": "MFH"
    },
    "Building_3": {
        "gross_area": 180,
        "type": "EFH"
    },
    "Building_4": {
        "gross_area": 650,
        "type": "MFH"
    },
    "Building_5": {
        "gross_area": 400,
        "type": "MFH"
    }
}

location = "Example Block"

original_apartment_sizes = {
    "Example Block": {
        '0-39': 12,
        '40-59': 20,
        '60-79': 3,
        '80-99': 3,
        '100-119': 0,
        '120-139': 7,
        '140-159': 4,
        '160-179': 1,
        '180-199': 0,
        '200-1000': 0
    }
}
original_apartments_per_building = {
    "Example Block": {
        '1': 0,
        '2': 1,
        '3-6': 0,
        '7-12': 3,
        '13-50000': 1
    }
}

def create_normal_distribution(apartment_sizes):
    sizes = []
    counts = []
    for size_range, count in apartment_sizes.items():
        if '-' in size_range:
            start, end = map(int, size_range.split('-'))
            mid = (start + end) / 2
        else:
            mid = int(size_range.split('-')[0])
        sizes.append(mid)
        counts.append(count)

    mean = np.average(sizes, weights=counts)
    std = np.sqrt(np.average((np.array(sizes) - mean) ** 2, weights=counts))
    return stats.norm(mean, std)

def calculate_probabilities(building, norm_dist, apartments_per_building):
    probabilities = {}
    gross_area = building['gross_area']
    building_type = building['type']

    # Calculate the maximum limit
    max_apartments = min(50000, gross_area // 20)  # Set an absolute upper limit of 50000

    for range_key, count in apartments_per_building.items():
        if range_key == '13-500':
            # For SFH (Single Family Home), we skip this range
            if building_type == 'EFH':
                continue
            # Handle the range from 13 to the calculated limit
            for num_apartments in range(13, max_apartments + 1):
                area_per_apartment = gross_area / num_apartments
                if num_apartments == 13:
                    prob = 1 - norm_dist.cdf(area_per_apartment)
                else:
                    next_area = gross_area / (num_apartments + 1)
                    prob = norm_dist.cdf(area_per_apartment) - norm_dist.cdf(next_area)

                probabilities[num_apartments] = prob * (count / sum(apartments_per_building.values()))
        elif '-' in range_key:
            start, end = map(int, range_key.split('-'))
            # For SFH, we limit to a maximum of 2 apartments
            if building_type == 'EFH':
                end = min(end, 2)
            # For MFH (Multi-Family Home), we start from 3 apartments
            elif building_type == 'MFH' and start < 3:
                start = 3
            range_values = range(start, min(end, max_apartments) + 1)
            for num_apartments in range_values:
                area_per_apartment = gross_area / num_apartments
                if num_apartments == start:
                    prob = 1 - norm_dist.cdf(area_per_apartment)
                else:
                    next_area = gross_area / (num_apartments + 1)
                    prob = norm_dist.cdf(area_per_apartment) - norm_dist.cdf(next_area)

                probabilities[num_apartments] = prob * (count / sum(apartments_per_building.values()))
        else:
            num_apartments = int(range_key)
            # For SFH, allow only 1 or 2 apartments
            if building_type == 'EFH' and num_apartments > 2:
                continue
            # For MFH, at least 3 apartments
            if building_type == 'MFH' and num_apartments < 3:
                continue
            area_per_apartment = gross_area / num_apartments
            prob = 1 - norm_dist.cdf(area_per_apartment)
            probabilities[num_apartments] = prob * (count / sum(apartments_per_building.values()))

    return probabilities

def distribute_apartments(buildings_data, location, original_apartment_sizes, original_apartments_per_building):
    norm_dist = create_normal_distribution(original_apartment_sizes[location])
    apartments_per_building = original_apartments_per_building[location].copy()
    results = {}
    probabilities = {}

    while buildings_data:
        if all(count == 0 for count in apartments_per_building.values()) or \
                (all(building['type'] == 'MFH' for building in buildings_data.values()) and
                 all(count == 0 for key, count in apartments_per_building.items() if
                     '-' in key and int(key.split('-')[0]) > 2)):
            apartments_per_building = original_apartments_per_building[location].copy()

        building_probabilities = {}
        for building_name, building_info in buildings_data.items():
            probs = calculate_probabilities(building_info, norm_dist, apartments_per_building)
            building_probabilities[building_name] = probs

        # Output the probabilities for all buildings
        print("\nProbabilities for all buildings:")
        for building, probs in building_probabilities.items():
            print(f"{building}:")
            for num_apartments, prob in probs.items():
                print(f"  {num_apartments} apartments: {prob:.4f}")

        # Find the building with the highest probability
        max_prob = -1
        max_building = None
        max_num_apartments = None

        for building, probs in building_probabilities.items():
            for num_apartments, prob in probs.items():
                if prob > max_prob:
                    max_prob = prob
                    max_building = building
                    max_num_apartments = num_apartments

        # Output of the best building
        print(f"\nBest building: {max_building}")
        print(f"Number of apartments: {max_num_apartments}")
        print(f"Probability: {max_prob:.4f}")

        # Update the results and remove the building from buildings_data
        results[max_building] = max_num_apartments
        probabilities[max_building] = max_prob
        del buildings_data[max_building]

        # Update apartments_per_building
        for range_key, count in apartments_per_building.items():
            if '-' in range_key:
                start, end = map(int, range_key.split('-'))
                if start <= max_num_apartments <= end:
                    apartments_per_building[range_key] = max(0, count - 1)
                    break
            elif int(range_key) == max_num_apartments:
                apartments_per_building[range_key] = max(0, count - 1)
                break

        print("\nRemaining buildings:", list(buildings_data.keys()))
        print("Updated apartments_per_building:", apartments_per_building)
        print("-" * 50)

    return results, probabilities

# Perform the distribution
final_results, final_probabilities = distribute_apartments(buildings_data, location, original_apartment_sizes,
                                                           original_apartments_per_building)

print("\nFinal distribution of apartments:")
for building, num_apartments in final_results.items():
    print(f"{building}: {num_apartments} apartments (Probability: {final_probabilities[building]:.4f})")
