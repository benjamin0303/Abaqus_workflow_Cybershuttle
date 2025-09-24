import os
import numpy as np

# Load the reference .inp file
with open('PlasticLug_Hard_00025.inp', "r") as file:
    inp_contents = file.readlines()

# Load the time values
time_values = np.loadtxt('time.txt')

# Load the amplitude data
amp_data = np.load('data_amp_5000.npy')


def create_modified_inp_file_final_format(ref_inp_contents, time_values, amp_values, num_sim_i, save_path):
    # Create pairs of time and amplitude values with the specified format
    # Exclude the last pair for custom formatting
    paired_values = "\n".join([f"{t:.2f}, {amp:.16e}," for t, amp in zip(time_values[:-1], amp_values[:-1])])
    # Add the last pair with the specified format
    paired_values += f"\n{time_values[-1]:.2f}, {amp_values[-1]:.16e}\n**\n**\n** MATERIALS"

    # Find the section corresponding to *Amplitude, name=V_E_U_amp
    amplitude_start_index = [i for i, line in enumerate(ref_inp_contents) if "*Amplitude, name=V_E_U_amp" in line][0]

    # Find the ending index of the amplitude section (look for the next section starting with *)
    amplitude_end_index = next((i for i, line in enumerate(ref_inp_contents[amplitude_start_index+1:]) if line.startswith('*')), None)
    amplitude_end_index += amplitude_start_index + 1  # Correct the index based on the starting position

    # Replace the amplitude table in the reference .inp file
    modified_contents = ref_inp_contents[:amplitude_start_index+1] + [paired_values] + ref_inp_contents[amplitude_end_index:]

    # Save the modified .inp file
    file_name = f"Job_{num_sim_i}.inp"
    with open(os.path.join(save_path, file_name), "w") as file:
        file.writelines(modified_contents)

    return file_name





# Path to save the modified .inp files
save_path = "generated_inps_2"

# Create directory if it doesn't exist
if not os.path.exists(save_path):
    os.makedirs(save_path)


generated_files_final_format = []
for num_sim_i in range(5000):
    file_name = create_modified_inp_file_final_format(inp_contents, time_values, amp_data[num_sim_i, :], num_sim_i, save_path)
    generated_files_final_format.append(file_name)
