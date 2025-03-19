#include <iostream>
#include <fstream>
#include <vector>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

int main() {
    try {
        // Simulate the semaphore (replace with actual semaphore logic if needed)
        bool ns3_ready = true;

        // Step 1: Read action from JSON file
        std::ifstream action_file("actions.json");
        if (!action_file.is_open()) {
            std::cerr << "Error: Could not open actions.json for reading" << std::endl;
            std::cout << "Debug: sem_post(this->ns3_ready) would be called here." << std::endl;
            return 1;
        }

        json action_json;
        action_file >> action_json;  // Read JSON data into action_json object

        // Ensure that the "actions" key exists and is an array
        if (!action_json.contains("actions") || !action_json["actions"].is_array()) {
            std::cerr << "Error: Invalid JSON format. 'actions' array not found." << std::endl;
            std::cout << "Debug: sem_post(this->ns3_ready) would be called here." << std::endl;
            return 1;
        }

        // Parse the action vector
        std::vector<int> action_vector;
        for (auto& action : action_json["actions"]) {
            int action_value = action.get<int>();
            if (action_value != 0 && action_value != 1) {
                std::cerr << "Error: Invalid action value (must be 0 or 1): " << action_value << std::endl;
                continue;  // Skip invalid actions
            }
            action_vector.push_back(action_value);
        }

        // Simulate the number of eNBs (replace with actual logic if needed)
        const uint32_t num_enbs = 4;  // Example: 3 eNBs

        // Ensure the action vector size matches the number of eNBs
        if (action_vector.size() != num_enbs) {
            std::cerr << "Error: Received action vector size does not match the number of eNBs" << std::endl;
            std::cout << "Debug: sem_post(this->ns3_ready) would be called here." << std::endl;
            return 1;
        }

        // Step 2: Simulate applying actions to eNB nodes
        std::vector<int> enb_power(num_enbs, 0);  // Simulate eNB power levels
        int active_power = 10;  // Example: Active power level

        for (uint32_t i = 0; i < num_enbs; i++) {
            enb_power[i] = (action_vector[i] == 0) ? 0 : active_power;
            std::cout << "eNB " << i << " power: " << enb_power[i] << std::endl;
        }

        std::cout << "Debug: Actions applied successfully." << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Exception while reading actions.json: " << e.what() << std::endl;
        std::cout << "Debug: sem_post(this->ns3_ready) would be called here." << std::endl;
        return 1;
    }

    return 0;
}
