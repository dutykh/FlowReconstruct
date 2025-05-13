import numpy as np
import os

def generate_data(a=0.1, L=4*np.pi, N=1000, output_dir='data'):
    """
    Generate N random sample points x in [0, L], compute
    eta = a * sin(x) and phi = sin^2(x) - cos^4(x),
    then save the results to output_dir/data.csv.
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Sample x uniformly and sort in increasing order
    x = np.random.uniform(0, L, size=N)
    x.sort()

    # Compute eta and phi
    eta = a * np.sin(x)
    phi = np.sin(x)**2 - np.cos(x)**4

    # Stack data columns and save as CSV
    data = np.column_stack((x, eta, phi))
    output_path = os.path.join(output_dir, 'data.csv')
    header = 'x,eta,phi'
    np.savetxt(output_path, data, delimiter=',', header=header, comments='')

    print(f'Data saved to {output_path}')

if __name__ == '__main__':
    generate_data()
