# Code created on Aug 16 2022
# Last Modified on April 23 2024

import numpy as np
import pandas as pd
from astropy.time import Time
import os, warnings, re, json, requests

MU = 1.32712440042e11 # Sun gravitational parameter (km^3/s^2)
EARTH_F = 0.003353 # Earth's flattening factor (dimensionless)
EARTH_RADIUS = 6378 # Earth's equatorial radius (km)
OBL = 84381.448/3600 # Standard Obliquity of J2000 epoch (degrees)

def handle_runtimewarning():
    """
    Suppresses RuntimeWarnings from the poliastro library.

    Args:
        None

    Returns:
        None
    """
    
    warnings.filterwarnings("error", category = RuntimeWarning)

def check_internet_connection():
    """
    Checks for an active internet connection.

    The function attempts to create a socket connection to a public DNS server 
    (8.8.8.8 on port 53) with a timeout of 3 seconds. If the connection is 
    successful, it indicates an active internet connection and the function 
    returns True. Otherwise, it returns False.

    Args:
        None

    Returns:
        bool: True if an internet connection is detected, False otherwise.
    """
  
    import socket
    try:
        socket.create_connection(("8.8.8.8", 53), timeout = 3)
        return True
    except OSError:
        pass
    return False

def read_ADES_format():
    """
    Reads an ADES PSV formatted text file and returns a pandas DataFrame.

    The function first prompts the user for the file name and checks if the file
    exists in a folder named 'data'. If the folder doesn't exist, it will be 
    created. If the file is not found, the user will be prompted to retry with a
    different file or exit the function.

    The function then reads the file and parses the header to define the column 
    names for the DataFrame. It checks for specific lines starting with either 
    "permID", "provID", "artSat", or "trkSub" to identify the header row. 
    Comments in the file are skipped.

    If the file format is valid, the function reads the data using thw read_csv
    pandas dunction, specifying the pipe delimiter ("|"), number of initial 
    comments to skip, column names, and skipping initial whitespace in each 
    column. It then drops unnecessary columns like 'trkSub', 'mode', etc. and 
    sorts the DataFrame by epoch ('obsTime' column).

    Args:
        None

    Returns:
        df (pandas DataFrame): Parsed data from the file. Returns None if the 
                               file is not found or the format is invalid.
    """
    
    # Set or create folder and filename to save results
    folder = "data"
    if not os.path.exists(folder):
        print("\nThe folder does not exist. It will be created...")
        os.makedirs(folder)
        print("Folder created. Place the data files inside of it.")
        input("If files have been added to folder, press Enter to continue...")
    # Prompt the user for the file name
    while True:
        print("\nADES (PSV formatted) text file name:")
        file_name = input("* ")
        if not file_name.lower().endswith(".txt"):
            file_name += ".txt"
        file = os.path.join(folder, file_name)
        if not os.path.exists(file):
            print("\nFile not found. Make sure file is in 'Data' folder.")
            input("If files have been added to folder, press Enter to continue...")
            while True:
                print("\nRetry with same or another file? (YES 'Y' or NO 'N')")
                retry = input("Answer: ").upper()
                if retry in ('N','NO'):
                    return None
                elif retry in ('Y','YES'):
                    break
                else:
                    print("ERROR: Answer must be YES (Y) or NO (N). Enter a valid answer.")
            if retry in ('Y','YES'):
                continue
        else:
            break
    # Read file and define column names for the DataFrame
    with open(file, "r") as text:
        print("\nImporting data...")
        comments = 0
        for line in text:
            if (line.startswith("#") or line.startswith("!")):
                comments += 1
            elif (line.startswith("permID") or line.startswith("provID") 
                  or line.startswith("artSat") or line.startswith("trkSub")):
                columns = [col.strip() for col in line.split("|")]
                comments += 1
                valid = True
                break
            else:
                print("INVALID FORMAT. Check the file.")
                valid = False
                break
    # Read the text file using pandas
    if valid is True:
        df = pd.read_csv(file, delimiter="|", skiprows = comments, names = columns, 
                     skipinitialspace=True)
        df = df.dropna(axis = 1, how = 'all')
        cols_to_drop = ['trkSub', 'mode', 'rmsFit', 'astCat', 'mag', 'rmsMag', 'band', 'photCat', 'photAp', 'logSNR', 'exp']
        for col in cols_to_drop:
          if col in df.columns:
            df.drop(columns = col, inplace = True)
        df.sort_values(by = 'obsTime', inplace = True)
        return df
    else:
        return None
'''
# THIS FUNCTION IS NOT DONE, initial clone of read_ADES_format
def read_MPC_format():
    # Set or create folder and filename to save results
    folder = "data"
    if not os.path.exists(folder):
        print("\nThe folder does not exist. It will be created...")
        os.makedirs(folder)
        print("Folder created. Place the data files inside of it.")
        input("If files have been added to folder, press Enter to continue...")
    # Prompt the user for the file name
    while True:
        print("\nMPC 80-column text file name:")
        file_name = input("* ")
        if not file_name.lower().endswith(".txt"):
            file_name += ".txt"
        file = os.path.join(folder, file_name)
        if not os.path.exists(file):
            print("\nFile not found. Make sure file is in 'Data' folder.")
            input("If files have been added to folder, press Enter to continue...")
            while True:
                print("\nRetry with same or another file? (YES 'Y' or NO 'N')")
                retry = input("Answer: ").upper()
                if retry in ('N','NO'):
                    return None
                elif retry in ('Y','YES'):
                    break
                else:
                    print("ERROR: Answer must be YES (Y) or NO (N). Enter a valid answer.")
            if retry in ('Y','YES'):
                continue
        else:
            break
    # Read file and fefine column names for the DataFrame
    with open(file, "r") as text:
        print("\nImporting data...")
        comments = 0
        for line in text:
            if (line.startswith("#") or line.startswith("!")):
                comments += 1
            elif (line.startswith("permID") or line.startswith("provID") 
                  or line.startswith("artSat") or line.startswith("trkSub")):
                columns = [col.strip() for col in line.split("|")]
                comments += 1
                valid = True
                break
            else:
                print("INVALID FORMAT. Check the file.")
                valid = False
                break
    # Read the text file using pandas
    if valid is True:
        df = pd.read_csv(file, delimiter="|", skiprows = comments, names = columns, 
                     skipinitialspace=True)
        return df
    else:
        return None
'''
def round_to_one(number):
    """
    Works with sigma_decimals() to round sigma to 1 when larger than 0.95.

    The function converts the input number to a scientific notation string with 
    three decimal places (".3e"). It then splits the string into mantissa 
    (significant digits) and exponent parts. The mantissa is rounded to one 
    significant digit using the following rule:
        - If the mantissa is greater than or equal to 9.5 (rounded to one 
          significant digit will be 10 or greater), the function returns 1 
          (representing the significant digits to substract).
        - Otherwise, the function returns 0 (representing no change).

    Args:
        number (float): The number to be analyzed.

    Returns:
        sf (int): The significant figures to remove (0 or 1)
    """
    
    str_value = f"{number:.3e}"
    parts = str_value.split('e')
    mantissa = float(parts[0].rstrip('0'))
    if mantissa >= 9.5:
        sf = 1
    else:
        sf = 0
    return sf

def sigma_decimals(sigma):
    """
    Calculates the number of decimals to display for a given standard deviation.

    The function determines the appropriate number of decimal places to display 
    for a standard deviation value based on its magnitude. For sigma values 
    greater than 1, the function returns 0, indicating that no decimal places 
    are necessary (integer representation).

    For sigma values less than or equal to 1, the function calculates the 
    number of decimal places to display using the following logic:
        1. It calculates the absolute value of the negative floor of the base-10 
           logarithm of sigma. This represents the number of leading zeros to 
           the left of the decimal point in sigma's scientific notation 
           representation.
        2. It subtracts the result of calling the `round_to_one` function with 
           sigma as input. This subtraction accounts for the significant digit 
           of sigma (rounded to one) when close to 1.0 (sigma > 0.95). The final
           result is the number of decimal places to display.

    Args:
        sigma (float): The standard deviation value.

    Returns:
        ndec (int): The number of decimal places to display for sigma.
    """
    
    if sigma > 1:
        ndec = 0
    else:
        ndec = np.abs(-np.floor(np.log10(sigma))) - round_to_one(sigma)
    return ndec

def equatorial_to_ecliptic(R, V, incl, raan, w, COP):
    """
    Converts a set of orbital elements from the equatorial reference frame to 
    the ecliptic reference frame.

    This function uses the equations given by (Sharaf et al., 2014).

    Args:
        R (numpy.ndarray): The position vector in the equatorial frame (km).
        V (numpy.ndarray): The velocity vector in the equatorial frame (km/s).
        incl (float): The inclination angle in degrees.
        raan (float): The right ascension of the ascending node in degrees.
        w (float): The argument of perihelion in degrees.
        COP (bool): Flag indicating whether to transform the elements (True) or 
        the state vectors (False).

    Returns:
        If COP is False:
            R_ec (numpy array): The position vector in the ecliptic frame (km).
            V_ec (numpy array): The velocity vector in the ecliptic frame (km/s).
        If COP is True:
            inc_ec (float): The inclination angle in the ecliptic plane (degrees).
            lan (float): The longitude of the ascending node in the ecliptic plane (degrees).
            w_ec (float): The argument of perihelion in the ecliptic plane (degrees).
    """
    
    if COP is False: # Transform the state vectors to the ecliptic plane
        R_ec = np.array([R[0],  R[1] * np.cos(np.radians(OBL)) + R[2] * np.sin(np.radians(OBL)),
                         -R[1] * np.sin(np.radians(OBL)) + R[2] * np.cos(np.radians(OBL))])
        V_ec = np.array([V[0],  V[1] * np.cos(np.radians(OBL)) + V[2] * np.sin(np.radians(OBL)),
                         -V[1] * np.sin(np.radians(OBL)) + V[2] * np.cos(np.radians(OBL))])
        return R_ec, V_ec
    else:
        # Define the transformation parameters
        n = -1
        x = np.sin(np.radians(incl)) * np.sin(np.radians(raan))
        y = np.sin(np.radians(incl)) * np.cos(np.radians(raan)) * np.cos(np.radians(OBL)) + n * np.cos(np.radians(incl)) * np.sin(np.radians(OBL))
        z = np.cos(np.radians(incl)) * np.cos(np.radians(OBL)) - n * np.sin(np.radians(incl)) * np.cos(np.radians(raan)) * np.sin(np.radians(OBL))
        q = - np.sin(np.radians(OBL)) * np.sin(np.radians(raan))
        t = - np.cos(np.radians(OBL)) * np.sin(np.radians(incl)) - n * np.sin(np.radians(OBL)) * np.cos(np.radians(incl)) * np.cos(np.radians(raan))
        # Transform the orbital elements to the ecliptic plane
        inc_ec = np.degrees(np.arctan2(np.sqrt(x**2 + y**2), z))
        lan = np.degrees(np.arctan2(x, y))
        w_ec = w - np.degrees(np.arctan2(-q, -t))
        # Apply quadrant restrictions
        inc_ec = inc_ec % 180
        lan = lan % 360
        w_ec = w_ec % 360
        return inc_ec, lan, w_ec
    
def jpl_horizons_sun_position(epochs):
    """
    Retrieves the Sun's position vectors for a list of epochs using JPL Horizons System API.

    This function fetches the Sun's position vectors in the International Celestial 
    Reference Frame (ICRF).

    Args:
        epochs (list of astropy.time.Time objects): UTC epochs for which the Sun's 
                                                   position are required.

    Returns:
        sun_pos (numpy array): Sun's position vector (x, y, z) in kilometers for 
                               the corresponding n epochs.
    """
    
    query_epochs = "\n".join([str(epochs.jd) for epoch in epochs])            
    # Request parameters
    request_dict = {'format' : 'text',
                    'EPHEM_TYPE' : 'VECTOR',
                    'OUT_UNITS' : 'KM-S',
                    'COMMAND' : 'Sun',
                    'CENTER' : '500@399',
                    'CSV_FORMAT' : 'YES',
                    'REF_PLANE' : 'FRAME',
                    'REF_SYSTEM' : 'ICRF',
                    'TP_TYPE' : 'ABSOLUTE',
                    'VEC_LABELS' : 'YES',
                    'OBJ_DATA' : 'NO',
                    'TLIST' : query_epochs,
                    'TIME_TYPE' : 'UT'
                   }   
    response = requests.get("https://ssd.jpl.nasa.gov/api/horizons.api", 
                            params = request_dict)
    vector_list = response.text
    vectors = np.array([])
    for line in vector_list.split("\n"):
        if line.startswith(tuple(map(str,range(10)))):
            vectors = np.append(vectors , [float(x) for x in line.split(",")[2:5]])
    sun_pos = vectors.reshape(-1,3)
    return sun_pos

def body_query(id):
    """
    Queries the JPL Small-Body Database for information about a specific body.

    This function retrieves basic information about an body from the JPL Small-Body 
    Database (SBDB) using its identification code.

    Args:
        id (str): The identification code of the body (e.g., "1423" or "1936 QM").

    Returns:
        number (str): The designation number of the body (e.g., "1423").
        name (str): The full name of the body if available (e.g., "1423 Jose (1936 QM)").
    """
    
    request_dict = {'sstr' : id}
    response = requests.get("https://ssd-api.jpl.nasa.gov/sbdb.api", params = request_dict)
    body = json.loads(response.text)
    number = body['object']['des']
    name = body['object']['fullname']
    return number, name

def sbdb_query(e, q_a, i, om, w, tol):
    """
    Queries the JPL Small-Body Database for bodies with matching orbital elements.

    This function searches the JPL Small-Body Database (SBDB) for bodies that 
    have orbital elements within a specified tolerance of the provided values.

    Args:
        e (float): The eccentricity of the orbit.
        q_a (float): The semi-major axis (elliptical) or periheloin distance (parabloic and hyperbloic).
        i (float): The inclination of the orbit in degrees.
        om (float): The longitude of the ascending node in degrees.
        w (float): The argument of perihelion in degrees.
        tol (float): The tolerance value for the orbital element comparisons.

    Returns:
        matches (pandas DataFrame): Matching bodies' data extracted from the SBDB 
                                    response. The DataFrame columns depend on the 
                                    specific orbit type:
                                        - epoch (str): Epoch of observation.
                                        - pdes (str): Provisional designation number.
                                        - full_name (str): Full name of the body (if available).
                                        - e (float): Eccentricity.
                                        - a (float): Semi-major axis (AU).
                                        - q (float): Perihelion distance (AU).
                                        - i (float): Inclination (degrees).
                                        - om (float): Longitude of the ascending node (degrees).
                                        - w (float): Argument of perihelion (degrees).
                                        - ma (float): Mean anomaly (degrees).
                                        - per (float): Orbital period (days).
                                        - n (float): Mean motion (degrees/day).
                                        - ad (float): Aphelion distance (AU).
                                        - tp (float): Time of perihelion passage (JD TDB).
    """
    
    e_min, e_max = e - tol, e + tol
    q_a_min, q_a_max = q_a - tol, q_a + tol
    i_min, i_max = i - tol, i + tol
    om_min, om_max = om - tol, om + tol
    w_min, w_max = w - tol, w + tol
    if e < 1: # Elliptical Orbit
        filters = f'''
        {{
          "AND": [
            "e|RG|{e_min}|{e_max}",
            "a|RG|{q_a_min}|{q_a_max}",
            "i|RG|{i_min}|{i_max}",
            "om|RG|{om_min}|{om_max}",
            "w|RG|{w_min}|{w_max}"
          ]
        }}
        '''
        request_dict = {'fields' : 'epoch,pdes,full_name,e,a,i,om,w,ma,per,n,q,ad,tp',
                        'sb-cdata': filters,
                        'full-prec' : 'TRUE'
                       }
    elif e >= 1 and e < (1 + 1e-12): # Parabolic Orbit
        filters = f'''
        {{
          "AND": [
            "e|RG|{e_min}|{e_max}",
            "q|RG|{q_a_min}|{q_a_max}",
            "i|RG|{i_min}|{i_max}",
            "om|RG|{om_min}|{om_max}",
            "w|RG|{w_min}|{w_max}"
          ]
        }}
        '''
        request_dict = {'fields' : 'epoch,pdes,full_name,e,q,i,om,w,tp',
                        'sb-cdata': filters,
                        'full-prec' : 'TRUE'
                       }
    else: # Hyerbolic Orbit
        filters = f'''
        {{
          "AND": [
            "e|RG|{e_min}|{e_max}",
            "q|RG|{q_a_min}|{q_a_max}",
            "i|RG|{i_min}|{i_max}",
            "om|RG|{om_min}|{om_max}",
            "w|RG|{w_min}|{w_max}"
          ]
        }}
        '''
        request_dict = {'fields' : 'epoch,pdes,full_name,e,q,i,om,w,ma,a,tp',
                        'sb-cdata': filters,
                        'full-prec' : 'TRUE'
                       }
    response = requests.get("https://ssd-api.jpl.nasa.gov/sbdb_query.api", params = request_dict)
    results = json.loads(response.text)
    df = pd.DataFrame(results['data'], columns = results['fields'])
    matches = pd.DataFrame(columns = df.columns)
    for index, row in df.iterrows():
        processed_values = []        
        for i, value in enumerate(row):
            if isinstance(value, str):
                if i == 1:
                    new_value = value
                elif value == '-':
                    new_value = value
                elif value.startswith("."):
                    new_value = float("0" + value)
                elif value.startswith("-."):
                    new_value = float(value[:1] + "0" + value[1:])
                elif any(c.isalpha() for c in value):
                    if value.startswith(" "):
                        value = value.lstrip()
                    new_value = value
                else:
                    new_value = float(value)
            processed_values.append(new_value)
        matches.loc[index] = processed_values
    if e >= 1 and e < (1 + 1e-12):
        matches.insert(df.columns.get_loc('w') + 1, 'ta', '-')
        matches.insert(df.columns.get_loc('w') + 2, 'ma', '-')
    else:
        matches.insert(df.columns.get_loc('w') + 1, 'ta', '-')
    return matches

def jpl_horizons_elements(id, TDB):
    """
    Retrieves orbital elements for a target body using the JPL Horizons API.

    This function fetches the orbital elements of a target body (e.g., asteroid)
    at a specified epoch (TDB) from the JPL Horizons System using the Astroquery 
    library.

    Args:
        id (str): The identification code of the target celestial body (e.g., "1423" or "1936 QM").
        epochs (list of astropy.time.Time objects): LTC TDB Epochs for which the 
                                                    orbital elements are desired.

    Returns:
        jpl_cop (numpy array): The orbital elements for the target body at the 
                               requested n epochs. The elements in the array are 
                               (depending on orbit type):
                                   - EC (float): Eccentricity.
                                   - A (float): Semi-major axis (AU).
                                   - IN (float): Inclination (degrees).
                                   - OM (float): Longitude of the ascending node (degrees).
                                   - W (float): Argument of perihelion (degrees).
                                   - TA (float): True anomaly (degrees).
                                   - MA (float): Mean anomaly (degrees).
                                   - PR (float): Orbital period (days).
                                   - N (float): Mean motion (degrees/day).
                                   - QR (float): Perihelion distance (AU).
                                   - AD (float): Aphelion distance (AU).
                                   - Tp (float): Time of perihelion passage (JD TDB).
    """
    
    from astroquery.jplhorizons import Horizons
    body = Horizons(id = id, location = 'Sun', epochs = TDB)
    elements = body.elements(refplane = 'ecliptic', tp_type = 'absolute')
    EC = elements['e'].filled(0).copy().data  # Eccentricity
    A = elements['a'].filled(0).copy().data  # Semi-major axis
    IN = elements['incl'].filled(0).copy().data  # Inclination
    OM = elements['Omega'].filled(0).copy().data  # Longitude of ascending node
    W = elements['w'].filled(0).copy().data  # Argument of perihelion
    TA = elements['nu'].filled(0).copy().data  # Mean anomaly
    MA = elements['M'].filled(0).copy().data  # True anomaly
    PR = elements['P'].filled(0).copy().data  # Orbit Period
    N = elements['n'].filled(0).copy().data  # Mean Motion
    QR = elements['q'].filled(0).copy().data  # Perihelion Distance
    AD = elements['Q'].filled(0).copy().data  # Aphelion Distance
    Tp = elements['Tp_jd'].filled(0).copy().data  # JD Time of Perihelion
    # Create elements array depending on orbit type
    if EC < 1:
        jpl_cop = np.array([EC, A, IN, OM, W, TA, MA, PR, N, QR, AD, Tp])
    elif EC >= 1 and EC < (1 + 1e-12):
        jpl_cop = np.array([EC, QR, IN, OM, W, TA, MA, Tp])
    else:
        jpl_cop = np.array([EC, QR, IN, OM, W, TA, MA, A, Tp])
    return jpl_cop

def jpl_horizons_ephemerides(id, site, epochs):
    """
    Retrieves ephemerides for a target at specified epochs.

    This function fetches the right ascension (RA), declination (Dec), and their 
    associated 3-sigma uncertainties for a target body (e.g., asteroid) at a list 
    of epochs from the JPL Horizons System API using the Astroquery library.

    Args:
        id (str): The identification code of the target body (e.g., "1423" or "1936 QM").
        site (numpy ndarray or dict): The observing site information. If a 1D 
                                      NumPy array of shape (1, 8) is provided, 
                                      the first element (site[0, 0]) is assumed 
                                      to be the site MPC code. Otherwise, a dictionary 
                                      with the following keys is expected:
                                          - 'lon' (float): Longitude of the site (degrees).
                                          - 'lat' (float): Latitude of the site (degrees).
                                          - 'elevation' (float): Elevation of the site (meters).
        epochs (list of astropy.time.Time objects): UTC Epochs for which the ephemerides are desired.

    Returns:
        eph.T (numpy array): Rphemerides for the target at the n epochs. The 
                             columns are:
                                 - Right Ascension (degrees)
                                 - Declination (degrees)
                                 - 1 sigma Right Ascension uncertainty (arcseconds)
                                 - 1 sigma Declination uncertainty (arcseconds)
    """
    
    from astroquery.jplhorizons import Horizons
    if site.shape[1] == 8:
        location = site[0, 0]
    else:
        location = {'lon': float(site[0, 0]), 'lat': float(site[0, 1]), 
                    'elevation': float(site[0, 2])/1000}
    body = Horizons(id = id, location = location, epochs = epochs.jd)
    ephemerides = body.ephemerides(refsystem = "ICRF", skip_daylight = True, extra_precision = True)
    RA = ephemerides['RA'].filled(0).copy().data  # Right Ascension
    Dec = ephemerides['DEC'].filled(0).copy().data  # Declination
    RA_3sigma = ephemerides['RA_3sigma'].filled(0).copy().data/3600/3  # Uncertainty in RA
    Dec_3sigma = ephemerides['DEC_3sigma'].filled(0).copy().data/3600/3 # Uncertainty in DeC
    eph = np.array([RA, Dec, RA_3sigma, Dec_3sigma])
    return eph.T

def mpc_obs_query(obs_code):
    """
    Queries the Minor Planet Center (MPC) observation codes for a specific MPC code.

    This function retrieves information about a specific MPC observatory code 
    from projectpluto.com or a local file if it exists.

    Args:
        obs_code (str): The MPC observation code to query (e.g., "W95").

    Returns:
        obs (numpy array): Observation information for the given MPC code. 
                           The columns are:
                               - Code (str): The MPC observation code.
                               - Longitude (float): Longitude of the observatory (degrees).
                               - Latitude (float): Latitude of the observatory (degrees).
                               - Altitude (float): Altitude of the observatory (degrees).
                               - rho_cos_phi and rho_sin_phi (float): Parallax constants.
                               - Region (str): Region of the location.
                               - Name (str): Name of the observatory.
    """
    
    from datetime import datetime
    date = datetime.today().strftime("%Y-%m-%d")
    # Set or create folder and filename to save results
    folder = "data"
    if not os.path.exists(folder):
        os.makedirs(folder)
    temp_file = os.path.join(folder, f"temp_{date}.txt")
    file = os.path.join(folder, "mpc_codes.txt")
    # Delete old temp files
    for filename in os.listdir(folder):
      if filename.startswith("temp_") and filename != temp_file:
        os.remove(os.path.join(folder, filename))
    if not os.path.exists(temp_file):
        response = requests.get("https://www.projectpluto.com/mpc_stat.txt")
        with open(temp_file, "w") as text:
          modified_lines = [line[3:] for line in response.text.splitlines()]
          text.write("\n".join(modified_lines))
        with open(temp_file, "r") as original_text:
          with open(file, "w") as modified_text:
            for line in original_text:
              modified_text.write(line)
    # specify the widths of each column
    widths = [(0,3), (3,15), (15,27), (27,38), (38,47), (47,58), (58,74), (74,-1)]
    # read the file and specify the column widths
    df = pd.read_fwf(file, colspecs = widths, header=0, skipfooter=1, 
                     names = ['Code', 'Longitude', 'Latitude', 'Altitude',
                              'rho_cos_phi', 'rho_sin_phi','Region', 'Name'])
    obs_query = df.loc[df['Code'] == obs_code]
    obs = obs_query.to_numpy()
    return obs  

def orbit_class(id):
    """
    Queries the JPL Small-Body Database for the orbit classification of a body.

    This function retrieves the orbital classification (e.g., Main Belt Asteroid) 
    for a target body from the JPL Small-Body Database 
    (SBDB) using its identification code.

    Args:
        id (str): The identification code of the body (e.g., "1423" or "1936 QM").

    Returns:
        orbit_class (str): The orbital classification of the body (e.g., "MBA").
    """
    
    request_dict = {'sstr' : id}
    response = requests.get("https://ssd-api.jpl.nasa.gov/sbdb.api", params = request_dict)
    body = json.loads(response.text)
    orbit_class = body['object']['orbit_class']
    return orbit_class

def get_input_data():
    """
    Parses the observation data according to the method selected by the user.

    This function offers the user three choices for inputting observation data 
    for orbit determination:

        1. ADES (PSV formatted) Text File: User selects a file containing 
           observations in the Astronomical Data Exchange Format (ADES) specifically 
           formatted in the Pipe-Separated Values (PSV) format.
        2. MPC 80-column Text File: (NOT IMPLEMENTED YET) User selects a file 
           containing observations in the Minor Planet Center (MPC) 80-column text 
           file format. This format is not currently supported by the function.
        3. Manually: User enters the observational data for each observation 
           separately through a series of prompts.

    The function performs the following steps based on the user's selection:

    1. ADES (PSV formatted) Text File:

        - Reads the data from the selected ADES PSV file using a pandas DataFrame.
        - Validates the data by checking if it contains a single object and if 
          required columns ('permID' or 'provID') are present. If not, it prompts 
          the user to retry with another file.
        - If there are more than 3 observation tracklets, it prompts the user to 
          select the 3 they want to use for Gauss' method. It validates the 
          indices provided by the user to ensure they are within the data bounds 
          and correspond to unique observations.
        - Extracts the observatories MPC codes (stations) from the data. For each 
          code, it queries the Minor Planet Center (MPC) observatory database using 
          the `mpc_obs_query` function to retrieve observatory information 
          (longitude, latitude, altitude, etc.).
        - Extracts the three observations data: epoch, RA, DEC, uncertanties
        - Constructs a pandas DataFrame containing the observation data and 
          corresponding observatories information.
    
    2. MPC 80-column Text File: NOTHING, NOT IMPLEMENTED YET.
          
    3. Manually: 

        - Prompts the user to enter the body's identifier (MPC ID, name, 
          designation, or 'U' for unknown). It then queries the JPL Horizons 
          database using the `body_query` function (function not defined here) 
          to retrieve the body's name if an MPC ID or designation is provided. 
          It validates the user input to ensure it follows the allowed format.
        - Prompts the user to enter the observation data for each of the 3 
          observations required for Gauss' method. The data includes:
            - Location (MPC code or observatory coordinates)
            - UTC observation time (ISOT format)
            - Right Ascension (degrees)
            - Right Ascension uncertainty (arcseconds)
            - Declination (degrees)
            - Declination uncertainty (rms DEC)
        - Validates the user input for locations (MPC code format or 
          coordinates), epochs (ISOT format), and numeric values for RA, DEC, 
          rms RA, and rms DEC. It checks for non-zero uncertainties and ensures 
          RA values are within the [0, 360] degrees range and DEC values are 
          within the [-90, 90] degrees range.
                  
    Args:
        None

    Returns:
        If data is None or if the user decides not to try again:
            None 
        If data is valid:
            obs_df (pandas DataFrame): Observatories information.
            id (str): The identifier (IAU number, name, or 'U') of the target body.
            name (str): The full name of the target body.
            RA (numpy array): Right Ascension values for each observation (degrees). 
            sRA (numpy array): Right Ascension uncertainties (arcseconds).
            dec (numpy array): Declination values for each observation (degrees).
            sdec (numpy array): Declination uncertainties (arcseconds).
            Rs (numpy array): Heliocentric observer position for each observation.
            epochs (list of astropy.time.Time objects): UTC epochs of each observation.
    """
    
    while True:
        print("\nSelect import format:")
        print("\n1. ADES (PSV formatted) Text File\n2. MPC 80-column Text File\n3. None, enter manually")
        input_type = input("\nAnswer (Number): ")
        if input_type == '1':  # ADES
            while True:
                data = read_ADES_format()
                if data is None:
                    return None               
                if data.iloc[:, 0].nunique() != 1 or len(data) < 3 or \
                    ('permID' not in data.columns and 'provID' not in data.columns):
                    if data.iloc[:, 0].nunique() != 1:
                        print("\nDifferent objects in file. Only one is supported.")
                    elif 'permID' not in data.columns and 'provID' not in data.columns:
                        print("\nMissing 'permID' or 'provID' column.")
                    else:
                        print(f"\nThere are {len(data)} tracklets:\n")
                        print(data[['obsTime','ra','dec','rmsRA','rmsDec']])
                        print("\nGauss' method NEEDS 3.")
                    while True:
                        print("\nRetry with another file? (YES 'Y' or NO 'N')")
                        retry = input("Answer: ").upper()
                        if retry in ('N','NO'):
                            return None
                        elif retry in ('Y','YES'):
                            break
                        else:
                            print("ERROR: Answer must be YES (Y) or NO (N). Enter a valid answer.")
                    if retry in ('Y','YES'):
                        continue
                else:
                    if len(data) > 3:
                        print(f"\nThere are {len(data)} tracklets:\n")
                        print(data[['obsTime','ra','dec','rmsRA','rmsDec']])
                        print("\nOnly 3 can be used for Gauss' method.")
                        while True:
                            indices = input("Enter the indices (comma-separated) you want to select: ")
                            try:
                                indices = np.array([int(index.strip()) for index in indices.split(",")])
                            except ValueError:
                                print("\nERROR: INVALID INPUT FORMAT. Enter INTEGER values separated by COMMAS.")
                                continue
                            if len(indices) != 3:
                                print("\nERROR: INVALID QUANTITY. Enter EXACTLY three indices.")
                                continue
                            elif len(np.unique(indices)) != 3:
                                print("\nERROR: REPEATED INDICES. Enter UNIQUE values.")
                                continue
                            for index in indices:
                                if index < 0 or index >= len(data):
                                    print(f"\nERROR: INDEX {index} is OUT OF BOUNDS. Enter VALID indices.")
                                    out = True
                                    continue
                                else:
                                    out = False
                            if out is True:
                                continue
                            else:
                                # Select and print the chosen rows
                                data = data.iloc[indices]
                                print("\nSelected data:")
                                print(data[['obsTime','ra','dec','rmsRA','rmsDec']])
                                print()
                                break
                    break
            id = data.iloc[0, 0]
            obs_list = np.empty((0, 8))
            for obs_code in data['stn']:
                obs_code = obs_code.strip()
                obs = np.array(mpc_obs_query(obs_code.upper()))
                obs_list = np.append(obs_list, obs, axis = 0)
            df_cloumns = ['Code', 'Longitude', 'Latitude', 'Altitude', 
                          'rho_cos_phi', 'rho_sin_phi', 'Region', 'Name']
            obs_df = pd.DataFrame(obs_list, columns = df_cloumns)
            break
        elif input_type == '2': # MPC 80-column
            print("\nNOT IMPLEMENTED YET. Select another file format.")
            continue
        elif input_type == '3': # Manually
            print("\nEnter the following Observational Data:")
            while True:
                # Prompt the user for a body identifier
                print("\n1. Body's MPC ID (Either its IAU Number, Name, Designation or UNKNOWN 'U'):")
                id = input("* ").upper()
                if id == 'U':
                    name = "Unknown Body"
                    break
                elif re.match(r'^[a-zA-Z0-9]+([ /-]?[a-zA-Z0-9]+)*$', id):
                    while True:
                        # Try to query the JPL Horizons database for the orbital elements
                        try:
                            number, name = body_query(id)
                        except Exception:
                            # If the query fails, print an error message and ask the user for a new identifier
                            print("\nERROR: INVALID IDENTIFIER. Enter a VALID identifier.")
                            print("If UNKNOWN enter 'U' or 'u'.")
                            answer1 = None
                            break
                        print(f"\n→ {name}")
                        print("Is this the correct body? (YES 'Y' or NO 'N')")
                        answer1 = input("Answer: ").upper()
                        if answer1 in ('Y','YES'):
                            id = number
                            break
                        elif answer1 in ('N','NO'):
                            break
                        else:
                            print("ERROR: Answer must be YES (Y) or NO (N). Enter a valid answer.")
                            continue
                    if answer1 in ('Y','YES'):
                        break
                    else:
                        continue
                else:
                    print("ERROR: Only ALPHANUMERIC characters are allowed. Enter a VALID identifier.")
            variables = ["Locations", "UTC Epochs", "RA", "rms RA", "DEC", "rms DEC"]
            units = ["MPC Codes or Coordinates [longitude,latitude,elevation]", "", 
                     "[decimal degrees]", "[arcseconds]", "[decimal degrees]", 
                     "[arcseconds]"]
            formats = ["MPC code or Coordinates [deg.ffffff,+/-deg.ffffff,m.fff]", 
                       "ISOT format: yyyy-mm-ddThh:mm:ss.fff", "deg.ffffff", "arcsec.fff", 
                       "deg.ffffff", "arcsec.fff"]
            values = {}  # Dictionary to store user-input values
            for i, (var, unit, formt) in enumerate(zip(variables, units, formats)):
                while True:
                    print(f"\n{i+2}. {var} {unit}")
                    print(f"   Value format: {formt}\n   3 values separated by commas")
                    value_str = input("\n* ")          
                    if var == "Locations" and '[' in value_str:
                        coords_list = value_str.strip('[]').split('],[')
                        value_list = [[float(val) for val in coord.split(',')] for coord in coords_list]
                    else:
                        value_list = [val.strip().upper() for val in value_str.split(",")]
                    # Check if the number of values entered is correct for each variable
                    if len(value_list) != 3:
                        print("ERROR: Invalid number of values. Enter exactly three values separated by commas.")
                        correct_format = False
                        continue
                    elif var == "Locations":
                        if '[' not in value_str: # Check if there are any invalid codes
                            invalid_codes = [v for v in value_list if not re.match("^[A-Z0-9][0-9]{2}$", v)]
                            if invalid_codes:  
                                correct_format = False
                                print("\nERROR: ONE OR MORE INVALID CODES. Enter a VALID MPC Observatory code.")
                                continue
                            else:
                                for j, value in enumerate(value_list):
                                    while True:
                                        try:
                                            obs = mpc_obs_query(value)
                                            if obs.size == 0:
                                                print("\nERROR: SITE DOESN'T EXIST. Enter a VALID MPC Observatory code.")
                                                answer2 = None
                                                break
                                            obs_code, region, obs_name = obs[0, 0], obs[0, 6], obs[0, 7]
                                        except Exception:
                                            print("\nERROR: INVALID CODE. Enter a VALID MPC Observatory code.")
                                            answer2 = None
                                        print(f"\n→ Site {j+1}: {obs_code} - {obs_name} ({region})")
                                        print("Is this the correct observatory? (YES 'Y' or NO 'N')")
                                        answer2 = input("Answer: ").upper()
                                        if answer2 in ('Y', 'YES'):
                                            correct_format = True
                                            break
                                        elif answer2 in ('N', 'NO'):
                                            while True:
                                                print("\nEnter the correct MPC code: ")
                                                new_value = input("* ").upper()
                                                if not re.match("^[A-Z0-9][0-9]{2}$", new_value):
                                                    print("\nERROR: INVALID CODE. Enter a VALID MPC Observatory code.")
                                                    continue                                            
                                                while True:
                                                    try:
                                                        new_obs = mpc_obs_query(new_value)
                                                        if new_obs.size == 0:
                                                            print("\nERROR: SITE DOESN'T EXIST. Enter a VALID MPC Observatory code.")
                                                            answer21 = None 
                                                            break
                                                        new_obs_code, new_region, new_obs_name = new_obs[0, 0], new_obs[0, 6], new_obs[0, 7]
                                                    except Exception:
                                                        print("\nERROR: INVALID CODE. Enter a VALID MPC Observatory code.")
                                                        answer21 = None
                                                    print(f"\n→ {new_obs_code} - {new_obs_name} ({new_region})")
                                                    print("Is this the correct observatory? (YES 'Y' or NO 'N')")
                                                    answer21 = input("Answer: ").upper()
                                                    if answer21 in ('Y','YES'):
                                                        value_list[j] = new_value # Update the value in the list at the current index
                                                        break
                                                    elif answer21 in ('N','NO'):
                                                        break
                                                    else:
                                                        print("ERROR: Answer must be YES (Y) or NO (N). Enter a valid answer.")
                                                        continue
                                                if answer21 in ('Y','YES'):
                                                    answer2 = 'Y'
                                                    break
                                                else:
                                                    continue
                                            if answer21 in ('Y','YES'):
                                                break
                                        else:
                                            print("ERROR: Answer must be YES (Y) or NO (N). Enter a valid answer.")
                                            continue
                                    if answer2 not in ('Y', 'YES'):  # If not confirmed as correct
                                        correct_format = False
                                        break
                        else: # Check if there are any invalid coordinates
                            if not all(all(isinstance(item, float) for item in inner_list) for inner_list in value_list):
                                print("\nERROR: INVALID COORDINATES. Coordinates must be all floats.")
                                correct_format = False
                                continue            
                            else:
                                for k, value in enumerate(value_list):
                                    while True:
                                        site_lon, site_lat, site_alt = value[0], value[1], value[2]
                                        print(f"\n→ Site {k+1} Coordinates: {site_lon},{site_lat} (at {site_alt} m)")
                                        print("Are these the correct coordinates? (YES 'Y' or NO 'N')")
                                        answer2 = input("Answer: ").upper()
                                        if answer2 in ('Y', 'YES'):
                                            correct_format = True
                                            break
                                        elif answer2 in ('N', 'NO'):
                                            while True:
                                                print("\nEnter the correct coordinates lon[deg],lat[deg],alt[m]: ")
                                                new_value_str = input("* ")
                                                new_value = [float(val.strip()) for val in new_value_str.split(",")]
                                                if len(new_value) != 3:
                                                    print("ERROR: Invalid number of values. Enter exactly three values separated by commas.")
                                                    continue
                                                new_site_lon, new_site_lat, new_site_alt = new_value[0], new_value[1], new_value[2]
                                                if not all(isinstance(item, float) for item in new_value):
                                                    print("\nERROR: INVALID COORDINATES. Enter VALID NUMERIC values: lon,lat,elev.")
                                                    continue
                                                while True:
                                                    print(f"\n→ Coordinates: {new_site_lon},{new_site_lat} (at {new_site_alt} m)")
                                                    print("Are these the correct coordinates? (YES 'Y' or NO 'N')")
                                                    answer21 = input("Answer: ").upper()
                                                    if answer21 in ('Y','YES'):
                                                        value_list[k] = new_value # Update the value in the list at the current index
                                                        break
                                                    elif answer21 in ('N','NO'):
                                                        break
                                                    else:
                                                        print("ERROR: Answer must be YES (Y) or NO (N). Enter a valid answer.")
                                                        continue
                                                if answer21 in ('Y','YES'):
                                                    answer2 = 'Y'
                                                    break
                                                else:
                                                    continue
                                            if answer21 in ('Y','YES'):
                                                break
                                        else:
                                            print("ERROR: Answer must be YES (Y) or NO (N). Enter a valid answer.")
                                            continue
                                    if answer2 not in ('Y', 'YES'):  # If not confirmed as correct
                                        correct_format = False
                                        break
                                correct_format = True
                    elif var == "UTC Epochs": # Check the epochs for isot format
                        try:
                            for value in value_list:
                                Time(value, format = 'isot', scale ='utc') # Attempt epoch parsing
                            correct_format = True
                        except ValueError:
                            print("\nERROR: INVALID EPOCHS. They must be in ISOT format yyyy-mm-ddThh:mm:ss.fff.")
                            correct_format = False
                            continue
                    else:
                        try: # Check if values are numeric
                            value_list = [float(val) for val in value_list] 
                            # Check for non-zero rms RA and rms DEC
                            if var in ["rms RA", "rms DEC"] and any(val == 0 for val in value_list):
                                print(f"\nERROR: {var} cannot be zero. Please enter non-zero values.")
                                correct_format = False
                                continue
                            if var == "RA" and any(val < 0 or val > 360 for val in value_list):
                                print("\nERROR: RA values must be in the [0, 360] range. Please enter valid values.")
                                correct_format = False
                                continue
                            if var == "DEC" and any(val < -90 or val > 90 for val in value_list):
                                print("\nERROR: DEC values must be in the [-90, 90] range. Please enter valid values.")
                                correct_format = False
                                continue
                            correct_format = True
                        except ValueError:
                            print(f"ERROR: Invalid {var} values. Enter numeric values.")
                            correct_format = False
                            continue
                    if correct_format is True:
                        values[var] = value_list # Store the values in the dictionary
                        break
            data = pd.DataFrame({
                "site": [np.array(loc) if isinstance(loc, list) 
                         else loc for loc in values["Locations"]],
                "obsTime": values["UTC Epochs"],
                "ra": values["RA"],
                "rmsRA": values["rms RA"],
                "dec": values["DEC"],
                "rmsDec": values["rms DEC"]
            })
            data.sort_values(by = 'obsTime', inplace = True)
            if isinstance(data['site'][0], str):
                obs_list = np.empty((0, 8))
                for obs_code in data['site']:
                    obs_code = str(obs_code).strip()
                    obs = np.array(mpc_obs_query(obs_code.upper()))
                    obs_list = np.append(obs_list, obs, axis = 0)
                df_cloumns = ['Code', 'Longitude', 'Latitude', 'Altitude', 
                              'rho_cos_phi', 'rho_sin_phi', 'Region', 'Name']
                obs_df = pd.DataFrame(obs_list, columns = df_cloumns)
                break
            else:
                site_list = np.empty((3, 3))
                for i, site_coords in enumerate(data['site']):
                    site_list[i] = np.array(site_coords)
                df_cloumns = ['Longitude', 'Latitude', 'Altitude']
                obs_df = pd.DataFrame(site_list, columns = df_cloumns)
                break
        else:
            print("\nERROR: Answer must be ADES (1), MPC 80 (2), or Manually (3). Enter a valid answer.")
    number, name = body_query(id)
    RA = data['ra'].values
    dec = data['dec'].values
    sRA = data['rmsRA'].values/3600
    sdec = data['rmsDec'].values/3600
    epochs_list = [epoch.strip() for epoch in data['obsTime'].values]
    epochs = []
    obs_rs = np.empty((3, 3))
    for i, (index, row) in enumerate(obs_df.iterrows()):
        lon = row['Longitude']
        lat = row['Latitude']
        alt = row['Altitude'] / 1000  # Convert altitude to kilometers
        epoch = Time(epochs_list[i], format = 'isot', scale = 'utc', location = (lon, lat))
        # Calculate Local Sidereal Time
        lst = np.radians(epoch.sidereal_time('mean', lon).deg)
        rx = (EARTH_RADIUS / np.sqrt(1 - (2 * EARTH_F - EARTH_F**2) * np.sin(np.radians(lat))**2) + alt) * np.cos(np.radians(lat)) * np.cos(lst)
        ry = (EARTH_RADIUS / np.sqrt(1 - (2 * EARTH_F - EARTH_F**2) * np.sin(np.radians(lat))**2) + alt) * np.cos(np.radians(lat)) * np.sin(lst)
        rz = (EARTH_RADIUS * (1 - EARTH_F)**2 / np.sqrt(1 - (2 * EARTH_F - EARTH_F**2) * np.sin(np.radians(lat))**2) + alt) * np.sin(np.radians(lat))
        # Geocentric observer position
        obs_rs[i] = np.array([rx, ry, rz])
        epochs.append(epoch)
    sun_rs = jpl_horizons_sun_position(epochs)
    # Heliocentic observer position
    Rs = obs_rs - sun_rs
    if input_type != '3':
        print("Data imported from file.")
    return obs_df, id, name, RA, sRA, dec, sdec, Rs, epochs

def gauss_method(ra1, ra2, ra3, dec1, dec2, dec3, Rs, epochs, Root):
    """
    Calculates the initial orbit of a body using Gauss' method.

    This function implements Gauss' method to determine the initial orbit by 
    calculating the state vectors of a body using three right ascension (RA), 
    declination (DEC), and observation times; as well as iteratively improve the
    states vectors until reaching convergence.
    
    This function uses the equations given by (Curtis, 2014).

    Args:
      ra1 (float): Right Ascension for the 1st observation (degrees).
      ra2 (float): Right Ascension for the 2nd observation (degrees).
      ra3 (float): Right Ascension for the 3rd observation (degrees).
      dec1 (float): Declination for the 1st observation (degrees).
      dec2 (float): Declination for the 2nd observation (degrees).
      dec3 (float): Declination for the 3rd observation (degrees).
      Rs (numpy array): Heliocentric observer's position for each observation.
      epochs (list of astropy.time.Time objects): LTC UTC observation times.
      Root (int, optional): Index of the desired positive root from the 
                             characteristic polynomial. This can be used to improve 
                             efficiency if a specific root is known to be valid. 
                             Defaults to None.

    Returns:
      state_vectors (numpy array): Position and velocity vectors of the body at 
                                     the middle time (km, km/s).
      rhos (numpy array): Slant ranges (distances) for each observation (km).
      root (int, optional): The index of the positive root used in the calculation 
                            (only returned if `Root` was None in the input).
    """
    
    #handle_runtimewarning()
    R1, R2, R3 = Rs[0], Rs[1], Rs[2]
    t1, t2, t3 = 0, (epochs[1] - epochs[0]).sec, (epochs[2] - epochs[0]).sec
    # Topocentric body position unit vectors (direction cosine vectors)
    rho1_unit = np.array([np.cos(np.radians(ra1))*np.cos(np.radians(dec1)), np.sin(np.radians(ra1))*np.cos(np.radians(dec1)), np.sin(np.radians(dec1))])
    rho2_unit = np.array([np.cos(np.radians(ra2))*np.cos(np.radians(dec2)), np.sin(np.radians(ra2))*np.cos(np.radians(dec2)), np.sin(np.radians(dec2))])
    rho3_unit = np.array([np.cos(np.radians(ra3))*np.cos(np.radians(dec3)), np.sin(np.radians(ra3))*np.cos(np.radians(dec3)), np.sin(np.radians(dec3))])
    # Times between observations
    tau1 = t1 - t2
    tau3 = t3 - t2
    # Observation arc
    tau = tau3 - tau1
    # Independent cross products among the direction cosine vectors
    p1 = np.cross(rho2_unit, rho3_unit)
    p2 = np.cross(rho1_unit, rho3_unit)
    p3 = np.cross(rho1_unit, rho2_unit)
    # Scalar triple product of the direction cosine vectors
    Do = np.dot(rho1_unit, p1)
    # Scalar triple products among the ps and Rs
    D = np.array([np.dot(R1, p1), np.dot(R1, p2), np.dot(R1, p3)])
    D = np.vstack((D, [np.dot(R2, p1), np.dot(R2, p2), np.dot(R2, p3)]))
    D = np.vstack((D, [np.dot(R3, p1), np.dot(R3, p2), np.dot(R3, p3)]))
    # Dot product of middle heliocentric observer's position and direction cosine vector
    E = np.dot(R2, rho2_unit)
    # A and B coefficents for middle slant range
    A = 1/Do * (-D[0,1]*tau3/tau + D[1,1] + D[2,1] * tau1/tau)
    B = 1/6/Do * (D[0,1]*(tau3**2 - tau**2)*tau3/tau + D[2,1] * (tau**2 - tau1**2)*tau1/tau)
    # Eighth-order polynomial coefficients
    a = -(A**2 + 2*A*E + np.linalg.norm(R2)**2)
    b = -2*MU*B*(A + E)
    c = -(MU*B)**2
    # Calculate the positive roots of the polynomial
    Roots = np.roots([1, 0, a, 0, 0, b, 0, 0, c])
    posroots = np.real(Roots[(np.isreal(Roots)) & (Roots > 0)])
    npositive = len(posroots)
    if npositive == 1:
        if Root is None:
            root = 0
            x = posroots[0]
        else:
            x = posroots[Root]
    else:
        if Root is None:
            i = 0
            while i < npositive:
                try:
                    root = i
                    x = posroots[i]
                    f1 = 1 - 1/2 * MU * tau1**2 / x**3
                    f3 = 1 - 1/2 * MU * tau3**2 / x**3
                    g1 = tau1 - 1/6 * MU * (tau1 / x)**3
                    g3 = tau3 - 1/6 * MU * (tau3 / x)**3
                    rho2 = A + MU * B / x**3
                    rho1 = 1 / Do * ((6 * (D[2,0] * tau1 / tau3 + D[1,0] * tau / tau3) * x**3 + MU * D[2,0] * (tau**2 - tau1**2) * tau1 / tau3) / (6 * x**3 + MU * (tau**2 - tau3**2)) - D[0,0])
                    rho3 = 1 / Do * ((6 * (D[0,2] * tau3 / tau1 - D[1,2] * tau / tau1) * x**3 + MU * D[0,2] * (tau**2 - tau3**2) * tau3 / tau1) / (6 * x**3 + MU * (tau**2 - tau1**2)) - D[2,2])
                    r1 = R1 + rho1 * rho1_unit
                    r2 = R2 + rho2 * rho2_unit
                    r3 = R3 + rho3 * rho3_unit
                    v2 = (-f3 * r1 + f1 * r3) / (f1 * g3 - f3 * g1)
                    r2_mag = np.linalg.norm(r2)
                    v2_mag = np.linalg.norm(v2)
                    vr2 = np.dot(v2, r2) / r2_mag
                    alpha = 2/r2_mag - v2_mag**2 / MU
                    x1 = kepler_U(tau1, r2_mag, vr2, alpha)    
                    x3 = kepler_U(tau3, r2_mag, vr2, alpha)
                    ff1, gg1 = f_and_g(x1, tau1, r2_mag, alpha)
                    ff3, gg3 = f_and_g(x3, tau3, r2_mag, alpha)
                    break
                except OverflowError:
                    print(f"OVERFLOW ENCOUNTERED: Root #{i+1} failed. Trying with the next one.\n")
                    i += 1
            else:
                print("None of the roots worked. GAUSS' METHOD FAILED.\n\n")
                return None
        else:
            x = posroots[Root]
    # Lagrange f coefficients at times t1 and t3
    f1 = 1 - 1/2 * MU * tau1**2 / x**3
    f3 = 1 - 1/2 * MU * tau3**2 / x**3
    # Lagrange g coefficients at times t1 and t3
    g1 = tau1 - 1/6 * MU * (tau1 / x)**3
    g3 = tau3 - 1/6 * MU * (tau3 / x)**3
    # Slant range at t2
    rho2 = A + MU * B / x**3
    # Slant range at t1 
    rho1 = 1 / Do * ((6 * (D[2,0] * tau1 / tau3 + D[1,0] * tau / tau3) * x**3 + MU * D[2,0] * (tau**2 - tau1**2) * tau1 / tau3) / (6 * x**3 + MU * (tau**2 - tau3**2)) - D[0,0])
    # Slant range at t3
    rho3 = 1 / Do * ((6 * (D[0,2] * tau3 / tau1 - D[1,2] * tau / tau1) * x**3 + MU * D[0,2] * (tau**2 - tau3**2) * tau3 / tau1) / (6 * x**3 + MU * (tau**2 - tau1**2)) - D[2,2])
    # Body position vectors
    r1 = R1 + rho1 * rho1_unit
    r2 = R2 + rho2 * rho2_unit
    r3 = R3 + rho3 * rho3_unit
    # Velocity vector at t2 
    v2 = (-f3 * r1 + f1 * r3) / (f1 * g3 - f3 * g1)
    # Improve the accuracy of the initial estimates
    # Initialize the iterative improvement loop and set error tolerance
    rho1_old = rho1
    rho2_old = rho2
    rho3_old = rho3
    diff1 = 1
    diff2 = 1
    diff3 = 1
    n = 0
    nmax = 1000
    tol = 1e-10
    # Iterative improvement loop:
    while ((diff1 > tol) and (diff2 > tol) and (diff3 > tol)) and (n < nmax):
        n = n + 1
        # Compute quantities required by universal kepler's equation
        r2_mag = np.linalg.norm(r2)
        v2_mag = np.linalg.norm(v2)
        vr2 = np.dot(v2, r2) / r2_mag
        alpha = 2/r2_mag - v2_mag**2 / MU
        # Solve universal Kepler's equation for universal anomalies x1 and x3           
        x1 = kepler_U(tau1, r2_mag, vr2, alpha)    
        x3 = kepler_U(tau3, r2_mag, vr2, alpha)
        # Calculate the Lagrange f and g coefficients at times tau1 and tau3
        ff1, gg1 = f_and_g(x1, tau1, r2_mag, alpha)
        ff3, gg3 = f_and_g(x3, tau3, r2_mag, alpha)
        # Update the f and g functions at times tau1 and tau3 by averaging old and new
        f1 = (f1 + ff1)/2
        f3 = (f3 + ff3)/2
        g1 = (g1 + gg1)/2
        g3 = (g3 + gg3)/2
        # c1 and c3 coefficients
        c1 =  g3/(f1*g3 - f3*g1)
        c3 = -g1/(f1*g3 - f3*g1)
        # Slant ranges
        rho1 = 1/Do * (-D[0][0] + 1/c1 * D[1][0] - c3/c1 * D[2][0])
        rho2 = 1/Do * (-c1 * D[0][1] + D[1][1] - c3 * D[2][1])
        rho3 = 1/Do * (-c1 / c3 * D[0][2] + 1/c3 * D[1][2] - D[2][2])
        # Position vectors
        r1 = R1 + rho1 * rho1_unit
        r2 = R2 + rho2 * rho2_unit
        r3 = R3 + rho3 * rho3_unit
        # Velocity vector at t2 
        v2 = np.array((-f3 * r1 + f1 * r3) / (f1 * g3 - f3 * g1))
        # Calculate differences upon which to base convergence
        diff1 = abs(rho1 - rho1_old)
        diff2 = abs(rho2 - rho2_old)
        diff3 = abs(rho3 - rho3_old)
        # Update the slant ranges:
        rho1_old = rho1
        rho2_old = rho2
        rho3_old = rho3
    # Return the state vectors and slant range for the central observation:
    r = r2
    v = v2
    state_vectors = np.vstack((r, v))
    rhos = np.array([rho1, rho2, rho3])
    if Root is None:
        return state_vectors, rhos, root
    else:
        return state_vectors
    
def orbital_elements(R, V, epoch):
    """
    Calculates the osculating ecliptic orbital elements of a body given its 
    states vectors (position and velocity) at a specific epoch.
    
    This function calls equatorial_to_ecliptic() to transfrom from the equatorial 
    reference plane to the ecliptic.
    
    Args:
      R (numpy array): Position vector of the body (km).
      V (numpy array): Velocity vector of the body (km/s).
      epoch (astropy.time.Time object): LTC UTC Epoch of the middle observation.

    Returns:
      cop (numpy array): The osculating eliptic orbital elements for the target body at the 
                             requested n epochs. The elements in the array are 
                             (depending on orbit type):
                                 - e (float): Eccentricity.
                                 - a (float): Semi-major axis (AU).
                                 - incl (float): Inclination (degrees).
                                 - lan (float): Longitude of the ascending node (degrees).
                                 - w (float): Argument of perihelion (degrees).
                                 - nu (float): True anomaly (degrees).
                                 - M (float): Mean anomaly (degrees).
                                 - per (float): Orbital period (days).
                                 - mn (float): Mean motion (degrees/day).
                                 - peri (float): Perihelion distance (AU).
                                 - aph (float): Aphelion distance (AU).
                                 - tp (float): Time of perihelion passage (JD TDB).
    """
    
    from astropy.time import TimeDelta
    EPS = 1.0e-6 # Threshold for an elliptical orbit
    # Calculate the state vectors magnitudes
    r = np.linalg.norm(R)
    v = np.linalg.norm(V)
    # Calculate the radial speed
    vr = np.dot(R, V) / r
    # Calculate the specific angular momentum
    H = np.cross(R, V)
    h = np.linalg.norm(H)
    # Calculate the node line vector and magnitude
    N = np.cross([0, 0, 1], H)
    n = np.linalg.norm(N)
    # Calculate the eccentricity vector and magnitude
    E = 1 / MU * (R * (v**2 - MU / r) - r * vr * V)
    e = np.linalg.norm(E)
    # Calculate the semi-major axis
    if e < 1 or e > (1 + 1e-12): # Elliptical and Hyperbolic orbit
        a = (h**2 / (MU * (1 - e**2)))
    # Calculate the inclinaton
    incl = np.degrees(np.arccos(H[2]/h))
    # Calculate the right ascention of the ascending node
    if incl != 0: # Inclined orbit
        raan = np.degrees(np.arccos(N[0]/n))
        if N[1] < 0:
            raan = 360 - raan
    else: # Equatorial orbit
        raan = 0
    # Calculate the argument of perihelion
    if incl != 0: # Inclined orbit
        if e > EPS: # Non-circular orbit
            w = np.degrees(np.arccos(np.dot(N, E)/n/e))
            if E[2] < 0:
                w = 360 - w
        else:  # Circular orbit
            w = 0
    else: # Equatorial orbit
        if e > EPS: # Non-circular orbit
            w = np.degrees(np.arccos(E[0]/e))
            if E[1] < 0:
                w = 360 - w
        else: # Circular orbit
            w = 0
    # Calculate the true anomaly
    if incl != 0: # Inclined orbit
        if e > EPS: # Non-circular orbit
            nu = np.degrees(np.arccos(np.dot(E, R)/e/r))
            if vr < 0:
                nu = 360 - nu
        else: # Circular orbit
            nu = np.degrees(np.arccos(np.dot(N, R)/n/r))
            if R[2] < 0:
                nu = 360 - nu
    else: # Equatorial orbit
        if e > EPS: # Non-circular orbit
            nu = np.degrees(np.arccos(np.dot(E, R)/e/r))
            if vr < 0:
                nu = 360 - nu
        else: # Circular orbit
            nu = np.degrees(np.arccos(R[0]/r))
            if R[1] < 0:
                nu = 360 - nu
    # Calculate the mean anomaly
    if e < 1: # Elliptical orbit
        M = np.degrees(np.arctan2(-np.sqrt(1 - e**2) * np.sin(np.radians(nu)), 
                                   - e - np.cos(np.radians(nu))) + np.pi 
                        - e * (np.sqrt(1 - e**2) * np.sin(np.radians(nu)) / 
                               (1 + e * np.cos(np.radians(nu)))))
    elif e >= 1 and e < (1 + 1e-12): # Parabolic orbit
        M = 1/2 * np.tan(np.radians(nu/2)) + 1/6 * np.tan(np.radians(nu/2))**3
    else: # Hyperbolic orbit
        F = 2 * np.atanh((np.sqrt(e - 1) / np.sqrt(e + 1)) * np.tan(np.radians(nu/2)))
        M = e * np.sinh(F) - F
    # Transform inc, raan and w to the ecliptic plane
    incl, lan, w = equatorial_to_ecliptic(R, V, incl, raan, w, COP = True)
    # Calculate other orbital paramenters for circular and elliptical orbits
    if e < 1:
        # Calculate the Orbital Period
        per = 2 * np.pi * np.sqrt(a**3 / MU)
        # Calculate the Mean Motion
        mn = 360 / per
        # Calculate the Perihelion Distance
        peri = a * (1 - e)
        # Calculate the Aphelion Distance
        aph = a * (1 + e)
        # Calculate the JD Epoch of Perihelion Passage
        if nu < 180:
            tp = (epoch.tdb - TimeDelta(M / mn, format = 'sec')).jd
        else:
            tp = (epoch.tdb + TimeDelta(per - M / mn, format = 'sec')).jd
        cop = np.array([e, a, incl, lan, w, nu, M, per, mn, peri, aph, tp])
        return cop    
    # Calculate others orbital paramenters for parabolic orbits
    elif e == 1:
        peri = h**2 / (2 * MU)
        if nu < 180:
            tp = (epoch.tdb - TimeDelta(M * h**3 / MU**2, format = 'sec')).jd
        else:
            tp = (epoch.tdb + TimeDelta(M * h**3 / MU**2, format = 'sec')).jd
        cop = np.array([e, peri, incl, lan, w, nu, M, tp])
        return cop
    # Calculate others orbital paramenters for hyperbolic orbits
    else:
        peri = a * (1 - e)
        if nu < 180:
            tp = (epoch.tdb - TimeDelta(M * h**3 / (MU**2 * (e**2 - 1)**(3/2)), format = 'sec')).jd
        else:
            tp = (epoch.tdb + TimeDelta(M * h**3 / (MU**2 * (e**2 - 1)**(3/2)), format = 'sec')).jd
        cop = np.array([e, peri, incl, lan, w, nu, M, a, tp])
        return cop

def kepler_U(dt, ro, vro, alpha):
    """
    Solves Kepler's universal equation for the universal anomaly.

    Args:
      dt (float): Time difference between two observations (s).
      ro (float): Initial distance of the body (km).
      vro (float): Initial radial velocity of the body (km/s).
      alpha (float): reciprocal of the semimajor axis (1/km).

    Returns:
      x (float): Universal anomaly (km^0.5)
      
    Raises:
         Warning message: If the maximum number of iterations is reached without
                          convergence, a warning message is printed.
    """
    
    # Set an error tolerance and a limit on the number of iterations:
    NMAX = 1000
    ERROR = 1.0e-10
    # Starting value for x:
    x = np.sqrt(MU) * abs(alpha) * dt
    # Iterate until until convergence occurs within the error tolerance:
    n = 0
    ratio = 1
    while abs(ratio) > ERROR and n <= NMAX:
        n = n + 1
        C = stumpffC(alpha * x**2)
        S = stumpffS(alpha * x**2)
        F = ro * vro / np.sqrt(MU) * x**2 * C + (1 - alpha*ro) * x**3 * S + ro * x - np.sqrt(MU) * dt
        dFdx = ro * vro / np.sqrt(MU) * x * (1 - alpha * x**2 * S) + (1 - alpha * ro) * x**2 * C + ro
        ratio = F / dFdx
        x = x - ratio
    #...Deliver a value for x, but report that NMAX was reached:
    if n > NMAX:
        print(f"* No. iterations of Kepler's equation = {n} *")
        print(f"F/dFdx = {F/dFdx}\n")
    return x

def stumpffC(z):
    """
    Calculates the Stumpff C function for a given value z.

    Args:
      z (float): Product of alpha and x^2 (dimensionless).

    Returns:
      c (float): The value of the Stumpff C function at z (dimensionless).
    """
    
    if z > 0:
        c = (1 - np.cos(np.sqrt(z))) / z
    elif z < 0:
        c = (1 - np.cosh(np.sqrt(-z))) / z
    else:
        c = 1/2
    return c

def stumpffS(z):
    """
    Calculates the Stumpff S function for a given value z.

    Args:
      z (float): Product of alpha and x^2 (dimensionless).

    Returns:
      s (float): The value of the Stumpff S function at z (dimensionless).
    """
    
    if z > 0:
        s = (np.sqrt(z) - np.sin(np.sqrt(z))) / (np.sqrt(z))**3
    elif z < 0:
        s = (np.sinh(np.sqrt(-z)) - np.sqrt(-z)) / (np.sqrt(-z))**3
    else:
        s = 1/6
    return s
    
def f_and_g(x, t, ro, alpha):
    """
    Calculates the Lagrange f and g coefficients in terms of universal anomaly.

    Args:
      x (float): Universal anomaly after time t (km^0.5).
      t (float): Time difference between two observations (s).
      ro (float): Initial distance of the body (km).
      a (float): reciprocal of the semimajor axis (1/km).

    Returns:
      f (float): Lagrange f coefficient at the given x (dimensionless).
      g (float): Lagrange g coefficient at the given x (s).
    """
    
    z = alpha * x**2
    # Lagrange f coefficient
    f = 1 - x**2 / ro * stumpffC(z)
    # Lagrange g coefficient
    g = t - 1 / np.sqrt(MU) * x**3 * stumpffS(z)
    return f, g

def fDot_and_gDot(x, r, ro, alpha):
    """
    Calculates the time derivatives of the Lagrange f and g coefficients.
    
    Args:
      x (float): Universal anomaly after time t (km^0.5).
      r (float): distance of the body after time t (km).
      ro (float): Initial distance of the body (km).
      a (float): reciprocal of the semimajor axis (1/km).

    Returns:
      fDot (float): Time derivative of the Lagrange f coefficient (1/s).
      gDot (float): Time derivative of the Lagrange g coefficient (dimensionless).
    """
    
    z = alpha * x**2
    fDot = np.sqrt(MU) / r / ro * (z * stumpffS(z) - 1) * x
    gDot = 1 - x**2 / r * stumpffC(z)
    return fDot, gDot

def plot_orbit(name, tdb, R, V, coe, db_coe, theme):
    """
    Plots the orbit of a body and refrence JPL Horizons or SMDB match orbits.

    Args:
        name (str): Full name of the body.
        tdb (astropy.time.Time object): LTC TDB Epoch of the orbit.
        R (numpy array): Position vector of the body at the epoch.
        V (numpy array): Velocity vector of the body at the epoch.
        coe (numpy array: Osculating ecliptic orbital elements  of the body.
        db_coe (numpy array): Orbital elements of the target body from JPL Horizons 
                              or of the SBDB matches.
        theme (strl): Color theme for the plot ("dark" or "light"). Defaults to "light".

    Returns:
        None
    """
    
    import matplotlib.pyplot as plt
    from matplotlib.style import context
    from astropy import units as u
    from poliastro.bodies import (Sun, Mercury, Venus, Earth, Mars, Jupiter, 
                                  Saturn, Uranus, Neptune)
    from poliastro.frames import Planes
    from poliastro.twobody import Orbit
    from poliastro.plotting import OrbitPlotter
    from poliastro.plotting.orbit.backends.matplotlib import Matplotlib2D
    warnings.filterwarnings("ignore", category=UserWarning)
    BODY_COLORS = {
        "Sun": "#ffcc00",
        "Mercury": "#8c8680",
        "Venus": "#e6db67",
        "Earth": "#2a7bd1",
        "Moon": "#999999",
        "Mars": "#cc653f",
        "Jupiter": "#bf8f5c",
        "Saturn": "#decf83",
        "Uranus": "#7ebec2",
        "Neptune": "#3b66d4",
    }
    # Set or create folder to save results
    folder = "results"
    if not os.path.exists(folder):
        os.makedirs(folder)
    # Assign State Vectors and COE to Orbit.from_classical inputs
    r = np.linalg.norm(R) << u.AU
    ecc = coe[0].item() << u.one
    if len(coe) >= 9:
        if len(coe) > 9:
            a = coe[1].item() << u.AU
        else:
            a = coe[7].item() << u.AU
    else:
        peri = coe[1].item() << u.AU
        l = 2 * peri 
    inc = coe[2].item() << u.deg
    lan = coe[3].item() << u.deg
    argp = coe[4].item() << u.deg
    nu = coe[5].item() << u.deg
    # Define the periapsis and ascending node directions
    periapsis = np.array([np.cos(np.radians(lan)) * np.cos(np.radians(argp)) 
                          - np.sin(np.radians(lan)) * np.sin(np.radians(argp)) * np.cos(np.radians(inc)),
                          np.sin(np.radians(lan)) * np.cos(np.radians(argp)) 
                          + np.cos(np.radians(lan)) * np.sin(np.radians(argp)) * np.cos(np.radians(inc))])
    ascending_node = np.array([np.cos(np.radians(lan)),np.sin(np.radians(lan))])
    # Define orbit epoch and reference plane
    epoch = Time(tdb, format = 'jd', scale = 'tdb')
    plane = Planes.EARTH_ECLIPTIC
    if theme == 'dark': # Dark Mode
        with context("dark_background"):
            plt.rcParams["font.family"] = "Arial"
            plt.rcParams["font.size"] = 12
            # Create a figure and axes
            fig, ax = plt.subplots(figsize=(12, 8))
            for i in range(3):
                # Save the orbit plots to folder
                if i == 0:
                    orbit = os.path.join(folder, f"{name} Orbit - Polar View.png")
                elif i == 1:
                    orbit = os.path.join(folder, f"{name} Orbit - Ecliptic Vernal View.png")
                else:
                    orbit = os.path.join(folder, f"{name} Orbit - Ecliptic Y-axis View.png")
                # Avoid drawing over the same figure
                ax.clear()
                # Plot body orbit views
                op = OrbitPlotter(backend = Matplotlib2D(ax = ax), plane = plane, length_scale_units = u.AU)
                if i == 0:
                    # Use "unit" orbit as the reference for plot
                    ref_coe = np.array([0, 1, 0, 0, 0, 0])
                    ref_orb = Orbit.from_classical(Sun, ref_coe[1]*u.AU, ref_coe[0]*u.one, ref_coe[2]*u.deg, 
                                                   ref_coe[3]*u.deg, ref_coe[4]*u.deg, ref_coe[5]*u.deg, 
                                                   epoch, plane)
                    ref = op.plot(ref_orb, label = "Epoch", color = 'k')
                    ref[1].set_linewidth(0)
                    ref[1].set_markersize(0)
                    ref[0].set_zorder(-1)
                    # Add the periapsis and ascending node directions
                    peri_label = "Periapsis Direction"
                    node_label = "Ascending Node Direction"
                    ax.arrow(0, 0, periapsis[0], periapsis[1], color = 'skyblue', lw = 0.5, head_width = 0.065, head_length = 0.065, length_includes_head = True, label = None, zorder = 1)
                    ax.arrow(0, 0, ascending_node[0], ascending_node[1], color = 'pink', lw = 0.5, head_width = 0.065, head_length = 0.065, length_includes_head = True, label = None, zorder = 1)
                    # Add the vernal (X) direction
                    vernal_label = "Vernal (X) Direction"
                    ax.arrow(0, 0, 1, 0, color = 'r', lw = 0.5, head_width = 0.065, head_length = 0.065, length_includes_head = True, label = None, zorder = 1)
                    # Add the Y-axis direction
                    y_label = "Y-axis Direction"
                    ax.arrow(0, 0, 0, 1, color = 'g', lw = 0.5, head_width = 0.035, head_length = 0.035, length_includes_head = True, label = None, zorder = 1)
                    # Set the title
                    ax.set_title(f"Orbit of {name} - Polar View", fontweight = "bold")
                    # Add the state vectors
                    pos_label = "Position Vector"
                    r_vec = ax.arrow(0, 0, R[0], R[1], color = 'y', lw = 0.5, head_width = 0.065, head_length = 0.065, length_includes_head = True, label = None, zorder = 1)
                    if np.linalg.norm(V) < 0.1:
                        vel_label = "Velocity Vector (25x scale)"
                        v_vec = ax.arrow(R[0], R[1], 25 * V[0], 25 * V[1], color = 'c', lw = 0.5, head_width = 0.065, head_length = 0.065, length_includes_head = True, label = None, zorder = 1)
                    else:
                        vel_label = "Velocity Vector"
                        v_vec = ax.arrow(R[0], R[1], V[0], V[1], color = 'c', lw = 0.5, head_width = 0.065, head_length = 0.065, length_includes_head = True, label = None, zorder = 1)
                    # Add items to legend
                    plt.scatter([], [], marker = r'$\rightarrow$', label = peri_label, c = 'skyblue', s = 150) # Periapsis directions
                    plt.scatter([], [], marker = r'$\rightarrow$', label = node_label, c = 'pink', s = 150) # Ascending node directions
                    plt.scatter([], [], marker = r'$\rightarrow$', label = vernal_label, c = 'r', s = 150) # Vernal (x) direction
                    plt.scatter([], [], marker = r'$\rightarrow$', label = y_label, c = 'g', s = 150) # Estival (y) direction
                    plt.scatter([], [], marker = r'$\rightarrow$', label = pos_label, c = 'y', s = 150) # Position vector
                    plt.scatter([], [], marker = r'$\rightarrow$', label = vel_label, c = 'c', s = 150) # Velocity vector
                elif i == 1:
                    # Use "unit" orbit as the reference for plot
                    ref_coe = np.array([0, r.to_value(u.AU)/3, 90, 90, 0, 0])
                    ref_orb = Orbit.from_classical(Sun, ref_coe[1]*u.AU, ref_coe[0]*u.one, ref_coe[2]*u.deg, 
                                                   ref_coe[3]*u.deg, ref_coe[4]*u.deg, ref_coe[5]*u.deg, 
                                                   epoch, plane)
                    ref = op.plot(ref_orb, label = "Epoch", color = 'k')
                    ref[1].set_linewidth(0)
                    ref[1].set_markersize(0)
                    ref[0].set_zorder(-1)
                    # Add the Y-axis direction
                    y_label = "Y-axis Direction"
                    ax.arrow(0, 0, 1, 0, color = 'g', lw = 0.5, head_width = 0.035, head_length = 0.035, length_includes_head = True, label = None, zorder = 1)
                    # Add the ecliptic pole (Z) direction
                    pole_label = "Ecliptic Pole (Z) Direction"
                    ax.arrow(0, 0, 0, 1, color = 'b', lw = 0.5, head_width = 0.035, head_length = 0.035, length_includes_head = True, label = None, zorder = 1)
                    # Set the title
                    ax.set_title(f"Orbit of {name} - Ecliptic Vernal View", fontweight = "bold")
                    # Add the state vectors
                    pos_label = "Position Vector"
                    r_vec = ax.arrow(0, 0, R[1], R[2], color = 'y', lw = 0.5, head_width = 0.05, head_length = 0.05, length_includes_head = True, label = None, zorder = 1)
                    if np.linalg.norm(V) < 0.1:
                        vel_label = "Velocity Vector (25x scale)"
                        v_vec = ax.arrow(R[1], R[2], 25 * V[1], 25 * V[2], color = 'c', lw = 0.5, head_width = 0.05, head_length = 0.05, length_includes_head = True, label = None, zorder = 1)
                    else:
                        vel_label = "Velocity Vector"
                        v_vec = ax.arrow(R[1], R[2], V[1], V[2], color = 'c', lw = 0.5, head_width = 0.05, head_length = 0.05, length_includes_head = True, label = None, zorder = 1)
                    # Add items to legend
                    plt.scatter([], [], marker = r'$\rightarrow$', label = y_label, c = 'g', s = 150) # Estival (y) direction
                    plt.scatter([], [], marker = r'$\rightarrow$', label = pole_label, c = 'b', s = 150) # Pole (z) direction
                    plt.scatter([], [], marker = r'$\rightarrow$', label = pos_label, c = 'y', s = 150) # Position vector
                    plt.scatter([], [], marker = r'$\rightarrow$', label = vel_label, c = 'c', s = 150) # Velocity vector
                else:
                    # Use "unit" orbit as the reference for plot
                    ref_coe = np.array([0, r.to_value(u.AU)/3, 90, 180, 0, 0])
                    ref_orb = Orbit.from_classical(Sun, ref_coe[1]*u.AU, ref_coe[0]*u.one, ref_coe[2]*u.deg, 
                                                   ref_coe[3]*u.deg, ref_coe[4]*u.deg, ref_coe[5]*u.deg, 
                                                   epoch, plane)
                    ref = op.plot(ref_orb, label = "Epoch", color = 'k')
                    ref[1].set_linewidth(0)
                    ref[1].set_markersize(0)
                    ref[0].set_zorder(-1)
                    # Add the vernal (X) direction
                    vernal_label = "Vernal (X) Direction"
                    ax.arrow(0, 0, -1, 0, color = 'red', lw = 0.5, head_width = 0.035, head_length = 0.035, length_includes_head = True, label = None, zorder = 1)
                    # Add the ecliptic pole (Z) direction
                    pole_label = "Ecliptic Pole (Z) Direction"
                    ax.arrow(0, 0, 0, 1, color = 'blue', lw = 0.5, head_width = 0.035, head_length = 0.035, length_includes_head = True, label = None, zorder = 1)
                    # Set the title
                    ax.set_title(f"Orbit of {name} - Ecliptic Y-axis View", fontweight = "bold")
                    # Add the state vectors
                    pos_label = "Position Vector"
                    r_vec = ax.arrow(0, 0, -R[0], R[2], color = 'y', lw = 0.5, head_width = 0.05, head_length = 0.05, length_includes_head = True, label = None, zorder = 1)
                    if np.linalg.norm(V) < 0.1:
                        vel_label = "Velocity Vector (25x scale)"
                        v_vec = ax.arrow(-R[0], R[2], -25 * V[0], 25 * V[2], color = 'c', lw = 0.5, head_width = 0.05, length_includes_head = True, head_length = 0.05, label = None, zorder = 1)
                    else:
                        vel_label = "Velocity Vector"
                        v_vec = ax.arrow(-R[0], R[2], -V[0], V[2], color = 'c', lw = 0.5, head_width = 0.05, head_length = 0.05, length_includes_head = True, label = None, zorder = 1)
                    # Add items to legend
                    plt.scatter([], [], marker = r'$\rightarrow$', label = vernal_label, c = 'r', s = 150) # Vernal (y) direction
                    plt.scatter([], [], marker = r'$\rightarrow$', label = pole_label, c = 'b', s = 150) # Pole (z) direction
                    plt.scatter([], [], marker = r'$\rightarrow$', label = pos_label, c = 'y', s = 150) # Position vector
                    plt.scatter([], [], marker = r'$\rightarrow$', label = vel_label, c = 'c', s = 150) # Velocity vector
                # Create Body Orbit
                if ecc < 1 or ecc > (1 + 1e-12): # Elliptical and Hyerbolic Orbit
                    orb = Orbit.from_classical(Sun, a, ecc, inc, lan, argp, nu, 
                                               epoch, plane)
                else: # Parabolic Orbit
                    orb = Orbit.parabolic(Sun, l, inc, lan, argp, nu, epoch, plane)
                ast = op.plot(orb, label = None, color = 'w', dashed = True)
                ast[0].set_linewidth(2)
                ast[0].set_zorder(3)
                ast[0].set_dashes([3, 3, 0, 1, 1, 0])
                ast[1].set_markersize(5)
                ast[1].set_markerfacecolor('purple')
                ast[1].set_markeredgecolor('purple')
                ast[1].set_zorder(4)
                # Add Body Orbit to the legend
                plt.plot([], [], label = 'Calculated Orbit', marker = 'o', 
                         ls = '--', c = 'w', mfc = 'purple', mec = 'purple')
                if name == "Unknown Body": # Use SBDB match orbit
                    sbdb_epoch = Time(db_coe[0], format = 'jd', scale='tdb')
                    sbdb_name = db_coe[2]
                    sbdb_ecc = db_coe[3] << u.one
                    if len(db_coe) >= 12:
                        if len(db_coe) > 12:
                            sbdb_a = db_coe[4].item() << u.AU
                        else:
                            sbdb_a = db_coe[10].item() << u.AU
                    else:
                        sbdb_peri = coe[4].item() << u.AU
                        sbdb_l = 2 * sbdb_peri                
                    sbdb_inc = db_coe[5] << u.deg
                    sbdb_lan = db_coe[6] << u.deg
                    sbdb_argp = db_coe[7] << u.deg
                    if len(db_coe) >= 12:
                        sbdb_ma = db_coe[9].item() << u.deg
                    else:
                        sbdb_ma = 0 << u.deg
                    if sbdb_ecc < 1 or sbdb_ecc > (1 + 1e-12): # Elliptical and Hyerbolic Orbit
                        sbdb_orb = Orbit.from_classical(Sun, sbdb_a, sbdb_ecc, sbdb_inc,
                                                       sbdb_lan, sbdb_argp, sbdb_ma, 
                                                       sbdb_epoch, plane)
                    else: # Parabolic Orbit
                        sbdb_orb = Orbit.parabolic(Sun, sbdb_l, sbdb_inc, sbdb_lan, 
                                                  sbdb_argp, sbdb_ma, sbdb_epoch, plane)
                    
                    sbdb = op.plot(sbdb_orb, label = None, color = 'g', 
                                   dashed = False)
                    sbdb[0].set_linestyle("-")
                    sbdb[1].set_markersize(8)
                    sbdb[1].set_markeredgecolor('c')
                    sbdb[1].set_zorder(3)
                    # Add SBDB Orbit to the legend
                    plt.plot([], [], label = (f"SBDB Orbit of {sbdb_name}" + 
                                              f"\nat epoch {sbdb_epoch.iso}"),
                             marker = 'o', ls = '-', c = 'g', mfc = 'g', 
                             mec = 'c')
                else: # Use Horizons orbit
                    jpl_ecc = db_coe[0].item() << u.one
                    if len(db_coe) >= 9:
                        if len(db_coe) > 9:
                            jpl_a = db_coe[1].item() << u.AU
                        else:
                            jpl_a = db_coe[7].item() << u.AU
                    else:
                        jpl_peri = coe[1].item() << u.AU
                        jpl_l = 2 * jpl_peri
                    jpl_inc = db_coe[2].item() << u.deg
                    jpl_lan = db_coe[3].item() << u.deg
                    jpl_argp = db_coe[4].item() << u.deg
                    jpl_nu = db_coe[5].item() << u.deg               
                    if jpl_ecc < 1 or jpl_ecc > (1 + 1e-12): # Elliptical and Hyerbolic Orbit
                        jpl_orb = Orbit.from_classical(Sun, jpl_a, jpl_ecc, jpl_inc,
                                                       jpl_lan, jpl_argp, jpl_nu, 
                                                       epoch, plane)
                    else: # Parabolic Orbit
                        jpl_orb = Orbit.parabolic(Sun, jpl_l, jpl_inc, jpl_lan, 
                                                  jpl_argp, jpl_nu, epoch, plane)
                    jpl = op.plot(jpl_orb, label = None, color = 'g', 
                                  dashed = False)
                    jpl[0].set_linestyle("-")
                    jpl[1].set_markersize(8)
                    jpl[1].set_markeredgecolor('c')
                    jpl[1].set_zorder(3)
                    # Add Horizons Orbit to the legend
                    plt.plot([], [], label = "JPL Horizons Orbit", marker = 'o', 
                             ls = '-', c = 'g', mfc = 'g', mec = 'c')
                # Plot orbits for the inner planets
                mercury_orb = op.plot_body_orbit(Mercury, epoch, label = None, trail = True)
                mercury_orb[1].set_markersize(4)
                venus_orb = op.plot_body_orbit(Venus, epoch, label = None, trail = True)
                venus_orb[1].set_markersize(4)
                earth_orb = op.plot_body_orbit(Earth, epoch, label = None, trail = True)
                earth_orb[1].set_markersize(4)
                # Add Inner Planets orbits to the legend
                plt.plot([], [], label = "Mercury", marker = 'o', ls = '-', c = BODY_COLORS['Mercury'])
                plt.plot([], [], label = "Venus", marker = 'o', ls = '-', c = BODY_COLORS['Venus'])
                plt.plot([], [], label = "Earth", marker = 'o', ls = '-', c = BODY_COLORS['Earth'])
                # Plot Mars orbit if needed
                if r.to_value(u.AU) > 1.15:
                    mars_orb = op.plot_body_orbit(Mars, epoch, label = None, trail = True)
                    mars_orb[1].set_markersize(4)
                    plt.plot([], [], label = "Mars", marker = 'o', ls = '-', c = BODY_COLORS['Mars'])
                # Plot orbits for the Outer Planets as needed
                if r.to_value(u.AU) > 3:
                    jupiter_orb = op.plot_body_orbit(Jupiter, epoch, label = None, trail = True)
                    jupiter_orb[1].set_markersize(4)
                    plt.plot([], [], label = "Jupiter", marker = 'o', ls = '-', c = BODY_COLORS['Jupiter'])
                if r.to_value(u.AU) > 7:
                    saturn_orb = op.plot_body_orbit(Saturn, epoch, label = None, trail = True)
                    saturn_orb[1].set_markersize(4)
                    plt.plot([], [], label = "Saturn", marker = 'o', ls = '-', c = BODY_COLORS['Saturn'])
                if r.to_value(u.AU) > 15:
                    uranus_orb = op.plot_body_orbit(Uranus, epoch, label = None, trail = True)
                    uranus_orb[1].set_markersize(4)
                    plt.plot([], [], label = "Uranus", marker = 'o', ls = '-', c = BODY_COLORS['Uranus'])
                if r.to_value(u.AU) > 25:
                    neptune_orb = op.plot_body_orbit(Neptune, epoch, label = None, trail = True)
                    neptune_orb[1].set_markersize(4)
                    plt.plot([], [], label = "Neptune", marker = 'o', ls = '-', c = BODY_COLORS['Neptune'])
                op.backend.update_legend()
                # Add a grid   
                ax.grid(c = 'gray', ls = '-', alpha = 0.25, zorder = 0)
                # Draw the Sun over the direction arrows
                sun_radius = op._attractor_radius.to_value(u.AU)
                sun = op.backend.draw_sphere(position = [0, 0, 0], color = BODY_COLORS['Sun'], label = None, radius = sun_radius)
                sun.set_zorder(1)
                r_vec.set_zorder(5)
                v_vec.set_zorder(5)
                if i == 0:
                    ax.set_xlabel("x [AU]", fontweight = 'bold')
                    ax.set_ylabel("y [AU]", fontweight = 'bold')
                elif i == 1:
                    ax.set_xlabel("y [AU]", fontweight = 'bold')
                    ax.set_ylabel("z [AU]", fontweight = 'bold')
                else:
                    tick_positions, tick_labels = plt.xticks()
                    inverted_tick_labels = tick_labels[::-1]
                    plt.xticks(tick_positions, inverted_tick_labels)
                    ax.set_xlabel("x [AU]", fontweight = 'bold')
                    ax.set_ylabel("z [AU]", fontweight = 'bold')
                # Save the orbit plots to folder
                plt.savefig(orbit, format = "png", dpi = 300, bbox_inches = 'tight')
    else: # Light Mode
        plt.rcParams["font.family"] = "Arial"
        plt.rcParams["font.size"] = 12
        # Create a figure and axes
        fig, ax = plt.subplots(figsize=(12, 8))
        for i in range(3):
            # Save the orbit plots to folder
            if i == 0:
                orbit = os.path.join(folder, f"{name} Orbit - Polar View.png")
            elif i == 1:
                orbit = os.path.join(folder, f"{name} Orbit - Ecliptic Vernal View.png")
            else:
                orbit = os.path.join(folder, f"{name} Orbit - Ecliptic Y-axis View.png")
            # Avoid drawing over the same figure
            ax.clear()
            # Plot body orbit views
            op = OrbitPlotter(backend = Matplotlib2D(ax = ax), plane = plane, length_scale_units = u.AU)
            if i == 0:
                # Use "unit" orbit as the reference for plot
                ref_coe = np.array([0, 1, 0, 0, 0, 0])
                ref_orb = Orbit.from_classical(Sun, ref_coe[1]*u.AU, ref_coe[0]*u.one, ref_coe[2]*u.deg, 
                                               ref_coe[3]*u.deg, ref_coe[4]*u.deg, ref_coe[5]*u.deg, 
                                               epoch, plane)
                ref = op.plot(ref_orb, label = "Epoch", color = 'w')
                ref[1].set_linewidth(0)
                ref[1].set_markersize(0)
                ref[0].set_zorder(-1)
                # Add the periapsis and ascending node directions
                peri_label = "Periapsis Direction"
                node_label = "Ascending Node Direction"
                ax.arrow(0, 0, periapsis[0], periapsis[1], color = 'skyblue', lw = 0.5, head_width = 0.065, head_length = 0.065, length_includes_head = True, label = None, zorder = 1)
                ax.arrow(0, 0, ascending_node[0], ascending_node[1], color = 'pink', lw = 0.5, head_width = 0.065, head_length = 0.065, length_includes_head = True, label = None, zorder = 1)
                # Add the vernal (X) direction
                vernal_label = "Vernal (X) Direction"
                ax.arrow(0, 0, 1, 0, color = 'r', lw = 0.5, head_width = 0.065, head_length = 0.065, length_includes_head = True, label = None, zorder = 1)
                # Add the Y-axis direction
                y_label = "Y-axis Direction"
                ax.arrow(0, 0, 0, 1, color = 'g', lw = 0.5, head_width = 0.035, head_length = 0.035, length_includes_head = True, label = None, zorder = 1)
                # Set the title
                ax.set_title(f"Orbit of {name} - Polar View", fontweight = "bold")
                # Add the state vectors
                pos_label = "Position Vector"
                r_vec = ax.arrow(0, 0, R[0], R[1], color = 'y', lw = 0.5, head_width = 0.065, head_length = 0.065, length_includes_head = True, label = None, zorder = 1)
                if np.linalg.norm(V) < 0.1:
                    vel_label = "Velocity Vector (25x scale)"
                    v_vec = ax.arrow(R[0], R[1], 25 * V[0], 25 * V[1], color = 'c', lw = 0.5, head_width = 0.065, head_length = 0.065, length_includes_head = True, label = None, zorder = 1)
                else:
                    vel_label = "Velocity Vector"
                    v_vec = ax.arrow(R[0], R[1], V[0], V[1], color = 'c', lw = 0.5, head_width = 0.065, head_length = 0.065, length_includes_head = True, label = None, zorder = 1)
                # Add items to legend
                plt.scatter([], [], marker = r'$\rightarrow$', label = peri_label, c = 'skyblue', s = 150) # Periapsis directions
                plt.scatter([], [], marker = r'$\rightarrow$', label = node_label, c = 'pink', s = 150) # Ascending node directions
                plt.scatter([], [], marker = r'$\rightarrow$', label = vernal_label, c = 'r', s = 150) # Vernal (x) direction
                plt.scatter([], [], marker = r'$\rightarrow$', label = y_label, c = 'g', s = 150) # Estival (y) direction
                plt.scatter([], [], marker = r'$\rightarrow$', label = pos_label, c = 'y', s = 150) # Position vector
                plt.scatter([], [], marker = r'$\rightarrow$', label = vel_label, c = 'c', s = 150) # Velocity vector
            elif i == 1:
                # Use "unit" orbit as the reference for plot
                ref_coe = np.array([0, r.to_value(u.AU)/3, 90, 90, 0, 0])
                ref_orb = Orbit.from_classical(Sun, ref_coe[1]*u.AU, ref_coe[0]*u.one, ref_coe[2]*u.deg, 
                                               ref_coe[3]*u.deg, ref_coe[4]*u.deg, ref_coe[5]*u.deg, 
                                               epoch, plane)
                ref = op.plot(ref_orb, label = "Epoch", color = 'w')
                ref[1].set_linewidth(0)
                ref[1].set_markersize(0)
                ref[0].set_zorder(-1)
                # Add the Y-axis direction
                y_label = "Y-axis Direction"
                ax.arrow(0, 0, 1, 0, color = 'g', lw = 0.5, head_width = 0.035, head_length = 0.035, length_includes_head = True, label = None, zorder = 1)
                # Add the ecliptic pole (Z) direction
                pole_label = "Ecliptic Pole (Z) Direction"
                ax.arrow(0, 0, 0, 1, color = 'b', lw = 0.5, head_width = 0.035, head_length = 0.035, length_includes_head = True, label = None, zorder = 1)
                # Set the title
                ax.set_title(f"Orbit of {name} - Ecliptic Vernal View", fontweight = "bold")
                # Add the state vectors
                pos_label = "Position Vector"
                r_vec = ax.arrow(0, 0, R[1], R[2], color = 'y', lw = 0.5, head_width = 0.05, head_length = 0.05, length_includes_head = True, label = None, zorder = 1)
                if np.linalg.norm(V) < 0.1:
                    vel_label = "Velocity Vector (25x scale)"
                    v_vec = ax.arrow(R[1], R[2], 25 * V[1], 25 * V[2], color = 'c', lw = 0.5, head_width = 0.05, head_length = 0.05, length_includes_head = True, label = None, zorder = 1)
                else:
                    vel_label = "Velocity Vector"
                    v_vec = ax.arrow(R[1], R[2], V[1], V[2], color = 'c', lw = 0.5, head_width = 0.05, head_length = 0.05, length_includes_head = True, label = None, zorder = 1)
                # Add items to legend
                plt.scatter([], [], marker = r'$\rightarrow$', label = y_label, c = 'g', s = 150) # Estival (y) direction
                plt.scatter([], [], marker = r'$\rightarrow$', label = pole_label, c = 'b', s = 150) # Pole (z) direction
                plt.scatter([], [], marker = r'$\rightarrow$', label = pos_label, c = 'y', s = 150) # Position vector
                plt.scatter([], [], marker = r'$\rightarrow$', label = vel_label, c = 'c', s = 150) # Velocity vector
            else:
                # Use "unit" orbit as the reference for plot
                ref_coe = np.array([0, r.to_value(u.AU)/3, 90, 180, 0, 0])
                ref_orb = Orbit.from_classical(Sun, ref_coe[1]*u.AU, ref_coe[0]*u.one, ref_coe[2]*u.deg, 
                                               ref_coe[3]*u.deg, ref_coe[4]*u.deg, ref_coe[5]*u.deg, 
                                               epoch, plane)
                ref = op.plot(ref_orb, label = "Epoch", color = 'w')
                ref[1].set_linewidth(0)
                ref[1].set_markersize(0)
                ref[0].set_zorder(-1)
                # Add the vernal (X) direction
                vernal_label = "Vernal (X) Direction"
                ax.arrow(0, 0, -1, 0, color = 'red', lw = 0.5, head_width = 0.035, head_length = 0.035, length_includes_head = True, label = None, zorder = 1)
                # Add the ecliptic pole (Z) direction
                pole_label = "Ecliptic Pole (Z) Direction"
                ax.arrow(0, 0, 0, 1, color = 'blue', lw = 0.5, head_width = 0.035, head_length = 0.035, length_includes_head = True, label = None, zorder = 1)
                # Set the title
                ax.set_title(f"Orbit of {name} - Ecliptic Y-axis View", fontweight = "bold")
                # Add the state vectors
                pos_label = "Position Vector"
                r_vec = ax.arrow(0, 0, -R[0], R[2], color = 'y', lw = 0.5, head_width = 0.05, head_length = 0.05, length_includes_head = True, label = None, zorder = 1)
                if np.linalg.norm(V) < 0.1:
                    vel_label = "Velocity Vector (25x scale)"
                    v_vec = ax.arrow(-R[0], R[2], -25 * V[0], 25 * V[2], color = 'c', lw = 0.5, head_width = 0.05, length_includes_head = True, head_length = 0.05, label = None, zorder = 1)
                else:
                    vel_label = "Velocity Vector"
                    v_vec = ax.arrow(-R[0], R[2], -V[0], V[2], color = 'c', lw = 0.5, head_width = 0.05, head_length = 0.05, length_includes_head = True, label = None, zorder = 1)
                # Add items to legend
                plt.scatter([], [], marker = r'$\rightarrow$', label = vernal_label, c = 'r', s = 150) # Vernal (y) direction
                plt.scatter([], [], marker = r'$\rightarrow$', label = pole_label, c = 'b', s = 150) # Pole (z) direction
                plt.scatter([], [], marker = r'$\rightarrow$', label = pos_label, c = 'y', s = 150) # Position vector
                plt.scatter([], [], marker = r'$\rightarrow$', label = vel_label, c = 'c', s = 150) # Velocity vector
            # Create Body Orbit
            if ecc < 1 or ecc > (1 + 1e-12): # Elliptical and Hyerbolic Orbit
                orb = Orbit.from_classical(Sun, a, ecc, inc, lan, argp, nu, 
                                           epoch, plane)
            else: # Parabolic Orbit
                orb = Orbit.parabolic(Sun, l, inc, lan, argp, nu, epoch, plane)
            ast = op.plot(orb, label = None, color = 'k', dashed = True)
            ast[0].set_linewidth(2)
            ast[0].set_zorder(3)
            ast[0].set_dashes([3, 3, 0, 1, 1, 0])
            ast[1].set_markersize(5)
            ast[1].set_markerfacecolor('purple')
            ast[1].set_markeredgecolor('purple')
            ast[1].set_zorder(4)
            # Add Body Orbit to the legend
            plt.plot([], [], label = 'Calculated Orbit', marker = 'o', 
                     ls = '--', c = 'w', mfc = 'purple', mec = 'purple')
            if name == "Unknown Body": # Use SBDB match orbit
                sbdb_epoch = Time(db_coe[0], format = 'jd', scale='tdb')
                sbdb_name = db_coe[2]
                sbdb_ecc = db_coe[3] << u.one
                if len(db_coe) >= 12:
                    if len(db_coe) > 12:
                        sbdb_a = db_coe[4].item() << u.AU
                    else:
                        sbdb_a = db_coe[10].item() << u.AU
                else:
                    sbdb_peri = coe[4].item() << u.AU
                    sbdb_l = 2 * sbdb_peri                
                sbdb_inc = db_coe[5] << u.deg
                sbdb_lan = db_coe[6] << u.deg
                sbdb_argp = db_coe[7] << u.deg
                if len(db_coe) >= 12:
                    sbdb_ma = db_coe[9].item() << u.deg
                else:
                    sbdb_ma = 0 << u.deg
                if sbdb_ecc < 1 or sbdb_ecc > (1 + 1e-12): # Elliptical and Hyerbolic Orbit
                    sbdb_orb = Orbit.from_classical(Sun, sbdb_a, sbdb_ecc, sbdb_inc,
                                                   sbdb_lan, sbdb_argp, sbdb_ma, 
                                                   sbdb_epoch, plane)
                else: # Parabolic Orbit
                    sbdb_orb = Orbit.parabolic(Sun, sbdb_l, sbdb_inc, sbdb_lan, 
                                              sbdb_argp, sbdb_ma, sbdb_epoch, plane)
                
                sbdb = op.plot(sbdb_orb, label = None, color = 'g', 
                               dashed = False)
                sbdb[0].set_linestyle("-")
                sbdb[1].set_markersize(8)
                sbdb[1].set_markeredgecolor('c')
                sbdb[1].set_zorder(3)
                # Add SBDB Orbit to the legend
                plt.plot([], [], label = (f"SBDB Orbit of {sbdb_name}" + 
                                          f"\nat epoch {sbdb_epoch.iso}"),
                         marker = 'o', ls = '-', c = 'g', mfc = 'g', 
                         mec = 'c')
            else: # Use Horizons orbit
                jpl_ecc = db_coe[0].item() << u.one
                if len(db_coe) >= 9:
                    if len(db_coe) > 9:
                        jpl_a = db_coe[1].item() << u.AU
                    else:
                        jpl_a = db_coe[7].item() << u.AU
                else:
                    jpl_peri = coe[1].item() << u.AU
                    jpl_l = 2 * jpl_peri
                jpl_inc = db_coe[2].item() << u.deg
                jpl_lan = db_coe[3].item() << u.deg
                jpl_argp = db_coe[4].item() << u.deg
                jpl_nu = db_coe[5].item() << u.deg               
                if jpl_ecc < 1 or jpl_ecc > (1 + 1e-12): # Elliptical and Hyerbolic Orbit
                    jpl_orb = Orbit.from_classical(Sun, jpl_a, jpl_ecc, jpl_inc,
                                                   jpl_lan, jpl_argp, jpl_nu, 
                                                   epoch, plane)
                else: # Parabolic Orbit
                    jpl_orb = Orbit.parabolic(Sun, jpl_l, jpl_inc, jpl_lan, 
                                              jpl_argp, jpl_nu, epoch, plane)
                jpl = op.plot(jpl_orb, label = None, color = 'g', 
                              dashed = False)
                jpl[0].set_linestyle("-")
                jpl[1].set_markersize(8)
                jpl[1].set_markeredgecolor('c')
                jpl[1].set_zorder(3)
                # Add Horizons Orbit to the legend
                plt.plot([], [], label = "JPL Horizons Orbit", marker = 'o', 
                         ls = '-', c = 'g', mfc = 'g', mec = 'c')
            # Plot orbits for the inner planets
            mercury_orb = op.plot_body_orbit(Mercury, epoch, label = None, trail = True)
            mercury_orb[1].set_markersize(4)
            venus_orb = op.plot_body_orbit(Venus, epoch, label = None, trail = True)
            venus_orb[1].set_markersize(4)
            earth_orb = op.plot_body_orbit(Earth, epoch, label = None, trail = True)
            earth_orb[1].set_markersize(4)
            # Add Inner Planets orbits to the legend
            plt.plot([], [], label = "Mercury", marker = 'o', ls = '-', c = BODY_COLORS['Mercury'])
            plt.plot([], [], label = "Venus", marker = 'o', ls = '-', c = BODY_COLORS['Venus'])
            plt.plot([], [], label = "Earth", marker = 'o', ls = '-', c = BODY_COLORS['Earth'])
            # Plot Mars orbit if needed
            if r.to_value(u.AU) > 1.15:
                mars_orb = op.plot_body_orbit(Mars, epoch, label = None, trail = True)
                mars_orb[1].set_markersize(4)
                plt.plot([], [], label = "Mars", marker = 'o', ls = '-', c = BODY_COLORS['Mars'])
            # Plot orbits for the Outer Planets as needed
            if r.to_value(u.AU) > 3:
                jupiter_orb = op.plot_body_orbit(Jupiter, epoch, label = None, trail = True)
                jupiter_orb[1].set_markersize(4)
                plt.plot([], [], label = "Jupiter", marker = 'o', ls = '-', c = BODY_COLORS['Jupiter'])
            if r.to_value(u.AU) > 7:
                saturn_orb = op.plot_body_orbit(Saturn, epoch, label = None, trail = True)
                saturn_orb[1].set_markersize(4)
                plt.plot([], [], label = "Saturn", marker = 'o', ls = '-', c = BODY_COLORS['Saturn'])
            if r.to_value(u.AU) > 15:
                uranus_orb = op.plot_body_orbit(Uranus, epoch, label = None, trail = True)
                uranus_orb[1].set_markersize(4)
                plt.plot([], [], label = "Uranus", marker = 'o', ls = '-', c = BODY_COLORS['Uranus'])
            if r.to_value(u.AU) > 25:
                neptune_orb = op.plot_body_orbit(Neptune, epoch, label = None, trail = True)
                neptune_orb[1].set_markersize(4)
                plt.plot([], [], label = "Neptune", marker = 'o', ls = '-', c = BODY_COLORS['Neptune'])
            op.backend.update_legend()
            # Add a grid   
            ax.grid(c = 'gray', ls = '-', alpha = 0.25, zorder = 0)
            # Draw the Sun over the direction arrows
            sun_radius = op._attractor_radius.to_value(u.AU)
            sun = op.backend.draw_sphere(position = [0, 0, 0], color = BODY_COLORS['Sun'], label = None, radius = sun_radius)
            sun.set_zorder(1)
            r_vec.set_zorder(5)
            v_vec.set_zorder(5)
            if i == 0:
                ax.set_xlabel("x [AU]", fontweight = 'bold')
                ax.set_ylabel("y [AU]", fontweight = 'bold')
            elif i == 1:
                ax.set_xlabel("y [AU]", fontweight = 'bold')
                ax.set_ylabel("z [AU]", fontweight = 'bold')
            else:
                tick_positions, tick_labels = plt.xticks()
                inverted_tick_labels = tick_labels[::-1]
                plt.xticks(tick_positions, inverted_tick_labels)
                ax.set_xlabel("x [AU]", fontweight = 'bold')
                ax.set_ylabel("z [AU]", fontweight = 'bold')
            # Save the orbit plots to folder
            plt.savefig(orbit, format = "png", dpi = 300, bbox_inches = 'tight')

def propagate_state_vectors(R0, V0, t):
    """
    Propagates the initial state vectors (position and velocity) of a body to a 
    new state at a specified time t in its orbit using universal Kepler's equation.

    Args:
      R0 (numpy array): The initial position vector of the body (km).
      V0 (numpy array): The initial velocity vector of the body (km/s).
      t (float): The time interval to propagate the state to (s).

    Returns:
      R (numpy array): The position vector of the body at time t (km).
      V (numpy array): The velocity vector of the body at time t (km/s).
    """
    
    # Magnitudes of R0 and V0:
    r0 = np.linalg.norm(R0)
    v0 = np.linalg.norm(V0)
    # Initial radial velocity:   
    vr0 = np.dot(R0, V0) / r0
    # Reciprocal of the semimajor axis (from the energy equation):
    alpha = 2 / r0 - v0**2 / MU
    # Compute the universal anomaly:
    x = kepler_U(t, r0, vr0, alpha)
    # Compute the f and g functions:
    f, g = f_and_g(x, t, r0, alpha)
    # Compute the final position vector:
    R = f * R0 + g * V0
    # Compute the magnitude of R:
    r = np.linalg.norm(R)
    # Compute the derivatives of f and g:
    fDot, gDot = fDot_and_gDot(x, r, r0, alpha)
    # Compute the final velocity:
    V = fDot * R0 + gDot * V0
    return R, V

def generate_ephemerides(start_epoch, ephemerides_epochs, R0s, V0s, n, ltc):
    """
    Generates light-Time corrected ephemerides (RA and DEC) with uncertanties 
    for a body at a specified epochs range.

    Args:
      start_epoch (astropy.time.Time object): The reference epoch.
      ephemerides_epochs (list of astropy.time.Time objects): A list of epochs at which 
                                                              to calculate ephemerides.
      R0s (numpy array): Array of N initial position vectors for the body.
      V0s (numpy array): Array of N initial velocity vectors for the body.
      n (int): The number of MC somulations.
      ltc (bool): Flag indicating whether dates include Light-Time Correction (True)
                  or not (False) and whether to calculate it.

    Returns:
      ephemerides (numpy array): Ephemerides for each epoch. Each row contains the 
                                 following variables:
                                     - np.mean(ra) (float): Mean Right Ascension (degrees)
                                     - np.mean(dec) (float): Mean Declination (degrees)
                                     - np.mean(delta): Mean slant range times (AU)
                                     - np.std(ra) (float): Standard deviation of right ascension (degrees)
                                     - np.std(dec) (float): Standard deviation of declination (degrees)
                                     - np.std(delta) (float): Standard deviation of slant range (AU)
                                     - mean_R[0] (float): Mean X component of position vector (AU)
                                     - mean_R[1] (float): Mean Y component of position vector (AU)
                                     - mean_R[2] (float): Mean Z component of position vector (AU)
                                     - mean_V[0] (float): Mean X component of velocity vector (AU/day)
                                     - mean_V[1] (float): Mean Y component of velocity vector (AU/day)
                                     - mean_V[2] (float): Mean Z component of velocity vector (AU/day)
    """
    
    if ltc is False: print()
    sun_rs = jpl_horizons_sun_position(ephemerides_epochs)
    ephemerides = np.zeros((len(ephemerides_epochs),12))
    for i, (eph_epoch, sun_r) in enumerate(zip(ephemerides_epochs, sun_rs)):
        t = (eph_epoch - start_epoch).sec
        Rs = np.zeros((n, 3))
        Vs = np.zeros((n, 3))
        ra = np.zeros(n)
        dec = np.zeros(n)
        delta = np.zeros(n)
        if ltc is False and i == 0:
            print('\rCalcutating Light-Time Correction (0.00%)', end = '', flush = True)
        for j in range(n):
            if ltc is True:
                print('\rCalculating ephemeris for epoch {} out of {} ({:.2f}%)'.format(i+1, len(ephemerides_epochs),(i+1)/len(ephemerides_epochs)*100), end = '', flush = True)
            sv = propagate_state_vectors(R0s[j], V0s[j], t)
            R = sv[0]
            V = sv[1]
            r = R + sun_r
            Rs[j] = R
            Vs[j] = V
            r_mag = np.sqrt(r[0]**2 + r[1]**2 + r[2]**2)
            unit_X = r[0] / r_mag
            unit_Y = r[1] / r_mag
            unit_Z = r[2] / r_mag
            decl = np.arcsin(unit_Z)
            if unit_Y > 0:
                ra[j] = np.degrees(np.arccos(unit_X / np.cos(decl)))
            else:
                ra[j] = 360 - np.degrees(np.arccos(unit_X / np.cos(decl)))
            dec[j] = np.degrees(decl)          
            delta[j] = r_mag / 149597870.7
        mean_R = np.mean(Rs, axis = 0) / 149597870.7
        mean_V = np.mean(Vs, axis = 0) * 0.000577548
        # Calculate the mean and std of the ephemerides
        eph = np.array([np.mean(ra), np.mean(dec), np.mean(delta), np.std(ra), np.std(dec), np.std(delta), mean_R[0], mean_R[1], mean_R[2], mean_V[0], mean_V[1], mean_V[2]])
        ephemerides[i] = eph
        if ltc is False:
            print('\rCalcutating Light-Time Correction ({:.2f}%)'.format((i+1)/len(ephemerides_epochs)*100), end = '', flush = True)
        if ltc is False and i == len(ephemerides_epochs) - 1:
            print()
    return ephemerides
