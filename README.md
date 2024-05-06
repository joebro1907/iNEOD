# NEIOD
A modern and open implementation of Gauss's method for the Initial Orbit Determination of Near-Earth Objects (Minor Planets)

## Description

The project is focused on the initial orbital determination for Near-Earth Objects (specifically, Asteroids). It involves functions for handling observational data, calculating positions, implementing Gauss's method for initial orbit determination, calculating orbital elements, solving Kepler's equation, and plotting orbits using Python and astronomy-related libraries.

## Usefulness

This project is valuable for students, enthusiasts, astronomers, and anyone interested in celestial mechanics. It provides tools and functions to analyze observational data, determine orbits of celestial objects, calculate orbital elements, and visualize orbits, which are essential tasks in the study of celestial bodies.

All this being open source.

## Installation and Use

1. The code requires the following Python libraries (tested on Python 3.11)

* Numpy
* Scipy
* Astropy
* Astroquery
* Matplotlib
* Pandas
* Poliastro (forked version: https://github.com/joebro1907/poliastro)

   You can install these libraries using [requirements.txt](https://github.com/joebro1907/NEIOD/blob/main/requirements.txt) in the source:

   `pip install -r requirements.txt`

2. Download this repository.

3. Read the docstrings on both [initial_orbit_determination.py](https://github.com/joebro1907/NEIOD/blob/main/initial_orbit_determination.py) and [iod_functions.py](https://github.com/joebro1907/NEIOD/blob/main/iod_functions.py) to understand how to use the functions and scripts effectively. This means understanding the input parameters, expected data formats, and the expected output.
  
   **_When running, the code itself will tell the you everything it needs and the expected format._**

4. Run the code with `initial_orbit_determination.py`

5. Test the code with the provided example data in the [data](https://github.com/joebro1907/NEIOD/tree/main/data) folder. Also you can try and modify some parameters like `nmax` or `tol` in the `gauss_method` function to see how the code behaves and how they can be adapted for your specific needs.

   You can also try modifying the input data to see how the accuracy changes, for example.

6. **Use you own data and calculate orbits!**

## Feedback and Collaboration

* Documentation: Both scripts should include detailed documentation explaining the purpose of each function, the expected inputs and outputs, and examples of usage.

   You can  contact me directly if you have specific questions not covered by the documentation.

* GitHub Repository Issues: If you find any bugs or would like to change/add things, you can use this repository's [issue tracker](https://github.com/joebro1907/NEIOD/issues) for known issues, discussions, and possible solutions.

## Maintenance

The project is likely to be maintained and contributed to by me and anoyone interested in celestial mechanics, astronomy, and Python programming. Users can reach out for support, contributions, or to report issues.

While the author of this repository (me, @joebro1907) is new somewhat new to Python, he will try his best :)
