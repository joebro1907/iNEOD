# Code created on Aug 16 2022
# Last Modified on April 23 2024

import warnings, os, re
import matplotlib.pyplot as plt
import numpy as np
from astropy.time import Time, TimeDelta
from astropy import units as u
from scipy.stats import norm
from iod_functions import (check_internet_connection, sigma_decimals,
                           get_input_data, equatorial_to_ecliptic,
                           jpl_horizons_elements, jpl_horizons_ephemerides,
                           sbdb_query, orbit_class, gauss_method, mpc_obs_query,
                           orbital_elements, generate_ephemerides, plot_orbit)

def initial_orbit_determination():
    """
    MAIN PROGRAM
    
    Performs initial orbit determination for a celestial body.

    This function takes observational data (RA and DEC (with uncertanties), and epochs) 
    and performs calculations to estimate the orbit of the body. 

    It includes the following steps:

      1. Data input and error handling.
      2. Unpacking observational data and setting up calculations.
      3. Slant range calculation and light-time correction.
      4. Generating N random samples based on measurement uncertainties.
      5. Initial orbit determination (state vectors) for N samples.
      6. Osculating ecliptic orbital elements calculation for N samples.
      7. Mean and standard deviation calculations for state vectors and elements.
      8. Astrometry, orbital elements histograms and orbit plotting.
      9. Saving results to a text file and plots.
      10. Optional orbit propagation for generating ephemerides. 
      
    Args:
      None (data is obtained through user prompts and file operations).

    Returns:
      None (function saves results to files and displays informative messages).
    """
    
    print("{:~^80}".format("# INITIAL ORBIT DETERMINATION #"))
    print("{:~^80}".format("| GAUSS\' METHOD |"))
    # Call the function to get the observational data
    input_data = get_input_data()
    # Terminate the program if data input was cancelled
    if input_data is None:
        return None
    else:
        (obs_df, id, name, ras, ra_stds, decs, dec_stds, obs_Rs, epochs) = input_data
    # Data sample size
    print("\nSet distribution sample size (at least 1 000)")
    print("NOTE: A sample size larger than 100 000 can take a long time to compute.\n")
    while True:
        try:
            N = int(input("Samples: "))
            if N < 1000:
                print(f"ERROR: {N} is NOT larger or equal to 1 000. Use a valid number")
            elif N >= 50000:
                print("Expect LARGE computing times with this sample size")
                break
            else:
                break
        except ValueError:
            print("ERROR: Sample size MUST be an INTEGER. Use a valid number")
    # Set or create folder and filename to save results
    folder = "results"
    if not os.path.exists(folder):
        os.makedirs(folder)
    file = os.path.join(folder, f"{name} - Results.txt")
    astrometry = os.path.join(folder, f"{name} - Astrometry.png")
    histogram = os.path.join(folder, f"{name} - Histograms.png")
    ephemerides = os.path.join(folder, f"{name} - Ephemerides.png")
    # Unpack all variables from input arrays
    ra1, ra2, ra3 = ras[0], ras[1], ras[2]
    ra1_std, ra2_std, ra3_std = ra_stds[0], ra_stds[1], ra_stds[2]
    dec1, dec2, dec3 = decs[0], decs[1], decs[2]
    dec1_std, dec2_std, dec3_std = dec_stds[0], dec_stds[1], dec_stds[2]
    # Start the calculation timer
    start_time1 = Time.now()
    # Get the slant ranges of the observations
    sv, rhos, root = gauss_method(ra1, ra2, ra3, dec1, dec2, dec3, obs_Rs, epochs, Root = None)
    ltcs = rhos / 299792.458
    ltc_epochs = np.array([])
    # Apply light time correction
    for i in range(len(epochs)):
        ltc_epochs = np.append(ltc_epochs, epochs[i] - TimeDelta(ltcs[i], format = 'sec'))
    ltc_epoch = ltc_epochs[1]
    # Generate N random RA and DEC values from STD of observations
    ras1 = np.random.normal(ra1, ra1_std, N)
    ras2 = np.random.normal(ra2, ra2_std, N)
    ras3 = np.random.normal(ra3, ra3_std, N)
    decl1 = np.random.normal(dec1, dec1_std, N)
    decl2 = np.random.normal(dec2, dec2_std, N)
    decl3 = np.random.normal(dec3, dec3_std, N)
    # Initialize State Vectors arrays
    r = np.zeros((N, 3))
    v = np.zeros((N, 3))
    # Initialize the Osculating Orbital Parameters arrays
    e = np.zeros(N)
    a = np.zeros(N)
    inc = np.zeros(N)
    o = np.zeros(N)
    w = np.zeros(N)
    ta = np.zeros(N)
    ma = np.zeros(N) 
    op = np.zeros(N)
    mn = np.zeros(N)
    peri = np.zeros(N)
    aph = np.zeros(N)
    tp = np.zeros(N)
    # Chech how many Orbital Parameters are there
    coe = orbital_elements(sv[0], sv[1], ltc_epoch)
    # Calculate the State Vectors and Orbital Parameters based on orbit type
    print("\nCalculating orbit...")   
    if len(coe) > 9: # Circular and Elliptical orbit
        for i in range(N):
            print('\rCalculating set {} out of {} ({:.2f}%)'.format(
                i+1, N, (i+1) / N * 100), end = '', flush = True)
            state_vectors = gauss_method(ras1[i], ras2[i], ras3[i], decl1[i], 
                                         decl2[i], decl3[i], obs_Rs, ltc_epochs, 
                                         Root = root)        
            if state_vectors is None:
                r[i, :] = v[i, :] = None
                e[i] = a[i] = inc[i] = o[i] = w[i] = ta[i] = ma[i] = None
                op[i] = mn[i] = peri[i] = aph[i] = tp[i] = None
            else:
                r[i, :] = state_vectors[0, :]
                v[i, :] = state_vectors[1, :]
                COP = orbital_elements(r[i, :], v[i, :], ltc_epoch)
                e[i] = COP[0]
                # Semi-major axis in astronomical units
                a[i] = COP[1] / 149597870.7
                inc[i] = COP[2]
                o[i] = COP[3]
                w[i] = COP[4]
                ta[i] = COP[5]
                ma[i] = COP[6]
                op[i] = COP[7] / 86400 # Orbital period in days
                mn[i] = COP[8] * 86400 # Mean motion in degrees/day
                # Perihelion distance in astronomical units
                peri[i] = COP[9] / 149597870.7
                # Aphelion distance in astronomical units
                aph[i] = COP[10] / 149597870.7
                tp[i] = COP[11]        
    elif len(coe) < 9: # Parabolic orbit
        for i in range(N):
            print('\rCalculating set {} out of {} ({:.2f}%)'.format(
                i+1, N, (i+1) / N * 100), end = '', flush = True)
            state_vectors = gauss_method(ras1[i], ras2[i], ras3[i], decl1[i], 
                                         decl2[i], decl3[i], obs_Rs, ltc_epochs, 
                                         Root = root)      
            if state_vectors is None:
                r[i, :] = v[i, :] = None
                e[i] = peri[i] = inc[i] = o[i] = w[i] = ta[i] = ma[i] = None
                tp[i] = None
            else:
                r[i, :] = state_vectors[0, :]
                v[i, :] = state_vectors[1, :]
                COP = orbital_elements(r[i, :], v[i, :], ltc_epoch)              
                e[i] = COP[0]
                # Perihelion distance in astronomical units
                peri[i] = COP[1] / 149597870.7
                inc[i] = COP[2]
                o[i] = COP[3]
                w[i] = COP[4]
                ta[i] = COP[5]
                ma[i] = COP[6]
                tp[i] = COP[7]
    else: # Hyperbolic orbit
        for i in range(N):
            print('\rCalculating set {} out of {} ({:.2f}%)'.format(
                i+1, N, (i+1) / N * 100), end = '', flush = True)
            state_vectors = gauss_method(ras1[i], ras2[i], ras3[i], decl1[i], 
                                         decl2[i], decl3[i], obs_Rs, ltc_epochs, 
                                         Root = root)      
            if state_vectors is None:
                r[i, :] = v[i, :] = None
                e[i] = peri[i] = inc[i] = o[i] = w[i] = ta[i] = ma[i] = None
                a[i] = tp[i] = None
            else:
                r[i, :] = state_vectors[0, :]
                v[i, :] = state_vectors[1, :]
                COP = orbital_elements(r[i, :], v[i, :], ltc_epoch)                
                e[i] = COP[0]
                # Perihelion distance in astronomical units
                peri[i] = COP[1] / 149597870.7
                inc[i] = COP[2]
                o[i] = COP[3]
                w[i] = COP[4]
                ta[i] = COP[5]
                ma[i] = COP[6]
                # Semi-major axis in astronomical units
                a[i] = COP[7] / 149597870.7
                tp[i] = COP[8]
    # Creating a mask to filter out None values
    mask = np.logical_not(np.any(np.isnan(r), axis = 1))
    r = r[mask]
    v = v[mask]
    if len(coe) > 9: # Circular and Elliptical orbit
        # Applying the mask to the arrays
        e = e[mask]
        a = a[mask]
        inc = inc[mask]
        o = o[mask]
        w = w[mask]
        ta = ta[mask]
        ma = ma[mask]
        op = op[mask]
        mn = mn[mask]
        peri = peri[mask]
        aph = aph[mask]
        tp = tp[mask]
    elif len(coe) < 9: # Parabolic orbit
        # Applying the mask to the arrays
        e = e[mask]
        peri = peri[mask]
        inc = inc[mask]
        o = o[mask]
        w = w[mask]
        ta = ta[mask]
        ma = ma[mask]
        tp = tp[mask]
    else: # Hyperbolic orbit
        # Applying the mask to the arrays
        e = e[mask]
        peri = peri[mask]
        inc = inc[mask]
        o = o[mask]
        w = w[mask]
        ta = ta[mask]
        ma = ma[mask]
        a = a[mask]
        tp = tp[mask]
    # Define new N as n from mask
    n = len(mask)
    # Transform the state vectors to the ecliptic plane
    r_ec = np.zeros((n, 3))
    v_ec = np.zeros((n, 3))
    for i in range(n):
        r_ec[i], v_ec[i] = equatorial_to_ecliptic(
            r[i], v[i], incl = 0, raan = 0, w = 0, COP = False)
    # Calculate the mean and std in AU
    mean_r_ec = np.array([np.mean(r_ec[:, 0]), np.mean(
        r_ec[:, 1]), np.mean(r_ec[:, 2])]) / 149597870.7
    sigma_r_ec = np.array([np.std(r_ec[:, 0]), np.std(
        r_ec[:, 1]), np.std(r_ec[:, 2])]) / 149597870.7
    mean_v_ec = np.array([np.mean(v_ec[:, 0]), np.mean(
        v_ec[:, 1]), np.mean(v_ec[:, 2])]) * 0.000577548
    sigma_v_ec = np.array([np.std(v_ec[:, 0]), np.std(
        v_ec[:, 1]), np.std(v_ec[:, 2])]) * 0.000577548
    # Stop the calculation timer
    end_time1 = Time.now()
    elapsed_time1 = end_time1 - start_time1
    # Tell the user is done calculating and tell the time it took
    if elapsed_time1.sec < 60:
        print(f"\nOrbit calculated in {elapsed_time1.sec:.3f}s")
        print("Orbit file saved to folder.")
    elif elapsed_time1.sec < 3600:
        m, s = divmod(elapsed_time1.sec, 60)
        print(f"\nOrbit calculated in {int(m)}m {int(s)}s")
        print("Orbit file saved to folder.")
    else:
        h, remainder = divmod(elapsed_time1.sec, 3600)
        m, s = divmod(remainder, 60)
        print(f"\nOrbit calculated in {int(h)}h {int(m)}m {int(s)}s")
        print("Orbit file saved to folder.")
    print("\nCreating Astrometry and Classical Orbital Elements plots...")
    # Start the first plotting timer
    start_time21 = Time.now()
    # Plot the astrometry measurements and errors
    plot_ras = np.array([ras1, ras2, ras3])
    plot_decs = np.array([decl1, decl2, decl3])
    plt.rcParams["font.family"] = "Arial"
    fig, axs = plt.subplots(2, 2, figsize = (12, 12))
    # Adapt data point transparency to the data density
    if N < 5000:
        alpha = 0.35
    elif N < 10000:
        alpha = 0.25
    else:
        alpha = 0.15
    # Plot the samples
    for i in range(3):
        row = i // 2
        col = i % 2
        axs[row, col].errorbar(ras[i], decs[i], xerr = ra_stds[i], yerr = dec_stds[i],
                        label = 'Measurement', marker = 's', c = 'r', alpha = 1,
                        zorder = 2)
        axs[row, col].scatter(plot_ras[i], plot_decs[i], label = 'Samples',
                       marker = 'o', c = '#0080FF', edgecolor = 'none', 
                       alpha = alpha, zorder = 1)
        axs[row, col].set_xlabel('α [deg]', fontweight = 'bold', fontsize = 12)
        axs[row, col].set_ylabel('δ [deg]', fontweight = 'bold', fontsize = 12)
        axs[row, col].set_title(f"Epoch: {ltc_epochs[i].iso}", fontweight = 'bold', fontsize = 14)
        axs[row, col].legend(fontsize = 12)
        axs[row, col].ticklabel_format(useOffset = False, style = 'plain')
        axs[row, col].xaxis.set_major_locator(plt.MaxNLocator(nbins = 5))
        axs[row, col].yaxis.set_major_locator(plt.MaxNLocator(nbins = 5))
        axs[row, col].grid(c = 'gray', ls = 'dashed', alpha = 0.25)
        axs[row, col].set_axisbelow(True)
    fig.delaxes(axs[1, 1])
    fig.suptitle(f'Astrometry of {name}', fontsize = 18, fontweight = 'bold')
    plt.tight_layout(rect = [0, 0, 1, 0.99])
    plt.savefig(astrometry, format = "png", dpi = 300) # Save the plots
    plt.close('all')
    # Create the figure and the subplots for the histograms
    plt.rcParams["font.family"] = "Arial"
    fig, ax = plt.subplots(nrows = 2, ncols = 3, figsize = (12, 8))
    # Plot the histograms and normal distribution curves
    if len(coe) > 9:
        # Define the data
        data = [e, a, inc, o, w, ta, ma, op, mn, peri, aph, tp]
        # Define the labels for the resutls
        labels = ["e", "a [AU]", "i [deg]", "om [deg]", "w [deg]", "nu [deg]", 
                  "M [deg]", "T [days]", "n [deg/day]", "q [AU]", "Q [AU]", 
                  "tp [JD]"]
        # Define the plot labels for the histograms
        titles = ["Eccentricity - e", "Semi-Major Axis - a [AU]", 
                  "Inclination - i [deg]", 
                  "Longitude of the Ascending Node - Ω [deg]",
                  "Argument of Perihelion - ω [deg]", 
                  r"True Anomaly - $\bf{ν}$ [deg]", "", "", "", "", "", ""]
    elif len(coe) < 9:
        # Define the data
        data = [e, peri, inc, o, w, ta, ma, tp]
        # Define the labels for the resutls
        labels = ["e", "q [AU]", "i [deg]", "om [deg]", "w [deg]", "nu [deg]", 
                  "M [deg]", "tp [JD]"]
        # Define the plot labels for the histograms
        titles = ["Eccentricity - e", "Perihelion Distance - q [AU]",
                  "Inclination - i [deg]", 
                  "Longitude of the Ascending Node - Ω [deg]",
                  "Argument of Perihelion - ω [deg]", 
                  r"True Anomaly - $\bf{ν}$ [deg]", "", ""]
    else:
        # Define the data
        data = [e, peri, inc, o, w, ta, ma, a, tp]
        # Define the labels for the resutls
        labels = ["e", "q [AU]", "i [deg]", "om [deg]", "w [deg]", "nu [deg]",
                  "M [deg]", "a [AU]", "tp [JD]"]
        # Define the plot labels for the histograms
        titles = ["Eccentricity - e", "Perihelion Distance - q [AU]",
                  "Inclination - i [deg]", 
                  "Longitude of the Ascending Node - Ω [deg]",
                  "Argument of Perihelion - ω [deg]", 
                  r"True Anomaly - $\bf{ν}$ [deg]", "", "", ""]
    mean_cop = np.empty((len(data),))
    sigma_cop = np.empty((len(data),))
    for i, (dat, title) in enumerate(zip(data, titles)):
        # Transform angles to the [-180,180] range in case the data is around 0
        if np.isin(dat, (inc, o, w, ta, ma)).any() and (np.any(dat <= 5) 
                                                        and np.any(dat >= 355)):
            dat = (dat + 180) % 360 - 180
        # Fit a normal distribution to the data and plot the curve
        mu, std = norm.fit(dat)
        mean_cop[i] = mu
        sigma_cop[i] = std
        if i < 6:
            # Plot the histogram
            ax[i // 3, i % 3].hist(dat, bins = 25, density = True,
                                   color = '#0080FF', edgecolor = '#0066CC')            
            xmin, xmax = ax[i // 3, i % 3].get_xlim()
            x = np.linspace(xmin, xmax, 100)
            p = norm.pdf(x, mu, std)
            ax[i // 3, i % 3].plot(x, p, 'r', linewidth = 2)
            # Transform back to the [0,360] range to display values
            if np.isin(dat, (inc, o, w, ta, ma)).any() and np.any(mu < 0):
                mu = mu + 360
                mean_cop[i] = mu
            # Add the title, the units and the text showing the mean and std
            ax[i // 3, i %
                3].set_title(f"{title}", fontweight = 'bold')
            ax[i // 3, i % 3].text(
                0.95, 0.95, f"µ = {mu:.3f}\nσ = {std:.3f}", ha = "right", 
                va = "top", transform = ax[i // 3, i % 3].transAxes, 
                fontweight = 'bold')
    fig.suptitle(f'Osculating Ecliptic Classical Orbital Elements of {name}', 
                     fontsize=14, fontweight='bold')
    plt.tight_layout() # Adjust the spacing between the subplots
    plt.savefig(histogram, format = "png", dpi = 300) # Save the histograms
    plt.close('all')
    # Stop the first plotting timer
    end_time21 = Time.now()
    elapsed_time21 = end_time21 - start_time21
    tdb = ltc_epoch.tdb.jd
    if id not in ('u', 'U'): # Compare to the orbital elements from JPL Horizons
        jpl_cop = jpl_horizons_elements(id, tdb)
        ape = np.empty(len(mean_cop))
        for i in range(len(mean_cop)):
            ape[i] = abs((jpl_cop[i, 0] - mean_cop[i]) / jpl_cop[i, 0]) * 100
    else: # Search JPL's Small-Body Database to compare possible match
        print("\nSearching JPL's Small-Body Database for a possible match...")  
        print("\nSet a tolerance between 0.1 and 1.0 for the matches (max 5):")
        print("NOTE: A tolenace larger than 2 result in too many matches")
        while True:
            try:
                tol = float(input("\nTolerance: "))
                if tol < 0.1:
                    print(f"ERROR: {tol} is NOT larger than 0.1." + 
                          " Use a valid number.\n")
                    continue
                if tol >= 2:
                    print("Expect TOO many matches")
                if tol > 5:
                        print(f"ERROR: Tolerance ({tol}) is TOO HIGH.\n")
                        print("Use a LOWER tolerance (maximum 5).\n")
                        continue
            except ValueError:
                print("ERROR: Value MUST be a number. Use a valid answer.\n")
                continue
            while True:
                print("\nSearching SBDB...")
                sbdb_matches = sbdb_query(mean_cop[0].item(), 
                                          mean_cop[1].item(), 
                                          mean_cop[2].item(), 
                                          mean_cop[3].item(),
                                          mean_cop[4].item(), tol)
                if len(sbdb_matches) == 0:
                    print("There were no matches found." + 
                          " Try again with a different tolerance?")
                    while True:
                        retry = input("Answer: ").upper()
                        if retry in ('Y', 'YES'):
                            break
                        if retry in ('N', 'NO'):
                            sbdb_matches = None
                            break
                        else:
                            print("ERROR: Answer must be YES (Y) or NO (N)." +
                                  " Enter a valid answer.")
                    break  
                elif len(sbdb_matches) > 1:
                    print("\nThere were more than one match. Select One:\n")
                    if mean_cop[0] < 1:
                        print(sbdb_matches[['full_name', 'e', 'a', 'i']])
                    else:
                        print(sbdb_matches[['full_name', 'e', 'q', 'i']])
                    while True:
                        try:
                            entry = int(input("\nNumber of match: "))
                            if entry < 0 or entry >= len(sbdb_matches):
                                print(f"ERROR: Index {entry} is out of bounds."
                                      + " Use a valid index.")
                            else:
                                sbdb_match = sbdb_matches.iloc[entry]
                                print(f"\n{sbdb_match['full_name']} selected.")
                                retry = 'N'
                                break
                        except ValueError:
                            print("ERROR: The index MUST be an INTEGER." + 
                                  " Use a valid number")
                    break
                else:
                    sbdb_match = sbdb_matches.iloc[0]
                    print(f"Match found: {sbdb_match['full_name']}\n")
                    retry = 'N'
                    break
            if retry not in ('Y', 'YES'):
                break
        sbdb_epoch = Time(sbdb_match[0], format = 'jd', scale = 'tdb')
        sbdb_id = sbdb_match[1]
        sbdb_name = sbdb_match[2]
        sbdb_cop = np.array(sbdb_match[3:], dtype = object)
        ape = np.empty(len(mean_cop), dtype = object)
        for i in range(len(mean_cop)):
            if i == 5 or i == 6:
                ape[i] = "-"
            else:
                ape[i] = (sbdb_cop[i] - mean_cop[i]) / sbdb_cop[i] * 100
    # Calculate the angular separation and observation arc
    ang_separation = np.degrees(np.arccos(np.sin(np.radians(decs[0])) * 
                                          np.sin(np.radians(decs[2])) + 
                                          np.cos(np.radians(decs[0])) * 
                                          np.cos(np.radians(decs[2])) * 
                                          np.cos(np.radians(ras[0]) - 
                                                 np.radians(ras[2]))))
    obs_arc = ltc_epochs[2] - ltc_epochs[0]
    # Get the otbit class
    if id not in ('u', 'U'):
        orbit_type = orbit_class(id)
    else:
        orbit_type = orbit_class(sbdb_id)
    orbit_code = orbit_type['code']
    orbit_name = orbit_type['name']
    # Save the results to a text file    
    with open(file, "w", encoding='utf-8') as text:
        text.write("{:-^80}".format("# Orbit of {} #".format(name)))
        text.write("\n\n{:~^80}\n".format(" Observational Data "))
        if len(obs_df.columns) == 8:
            if obs_df['Code'].nunique() == 1:
                text.write(f"\nObservatory: {obs_df.iloc[0, 0]} - {obs_df.iloc[0, 6]} ({obs_df.iloc[0, 7]})")
            else:
                text.write("\nObservatories (sorted by observation):")
                for i, obs in obs_df.iterrows():
                    obs_code, region, obs_name = obs[0], obs[6], obs[7]
                    text.write(f"\n{i+1}. {obs_code} - {obs_name} ({region})")
        else:
            if obs_df['Longitude'].nunique() == 1:
                lon, lat, alt = ((obs_df.iloc[0, 0] + 180) % 360 - 180), obs_df.iloc[0, 1], obs_df.iloc[0, 2]
                lat_str = str(lat + ' N') if lat >= 0 else str(abs(lat)) + ' S'
                lon_str = str(lon + ' E') if lon >= 0 else str(abs(lon)) + ' W'
                text.write(f"\nLocation: {lat_str}, {lon_str} (at {alt * 1000} m)")
            else:
                text.write("\nLocations (sorted by observation):")
                for i, obs in obs_df.iterrows():
                    lon, lat, alt = ((obs[0] + 180) % 360 - 180), obs[1], obs[2]
                    lat_str = str(lat + ' N') if lat >= 0 else str(abs(lat)) + ' S'
                    lon_str = str(lon + ' E') if lon >= 0 else str(abs(lon)) + ' W'
                    text.write(f"\n{i+1}. {lat_str}, {lon_str} (at {alt * 1000} m)")
        text.write(f"\n\nBody: {name}\n")
        text.write(("\n{:<31}{:<29}{:<20}".format("    Epoch (LTC UTC)", 
                                                  "RA +/- std [deg]", 
                                                  "DEC +/- std [deg]\n")))
        ra_stds_om = [sigma_decimals(ra_std) for ra_std in ra_stds]
        dec_stds_om = [sigma_decimals(dec_std) for dec_std in dec_stds]
        ra_values = [f'{ra_value:.{int(om)}f}' for ra_value, om in zip(ras, ra_stds_om)]
        ra_sigmas = [f'{ra_std:.{int(om)}f}' for ra_std, om in zip(ra_stds, ra_stds_om)]
        dec_values = [f'{dec_value:.{int(om)}f}' for dec_value, om in zip(decs, dec_stds_om)]
        dec_sigmas = [f'{dec_std:.{int(om)}f}' for dec_std, om in zip(dec_stds, dec_stds_om)]
        for i in range(len(ra_values)):
            text.write(("\n{:<26}{:>11} +/- {:<12}{:>12} +/- {:<9}"
                        .format(str(ltc_epochs[i].isot), ra_values[i], ra_sigmas[i],
                                dec_values[i], dec_sigmas[i])))
        text.write(f"\n\nAngular Separation (θ) [degrees] = {ang_separation:.3f}")
        text.write(f"\nObservation Arc [days] = {obs_arc.to_value(u.d):.3f}")
        text.write("\n\n{:~^80}\n".format(" Results "))
        if n < N:
            text.write(f"\nSamples: {n} successful (out of {N})")
        else:
            text.write(f"\nSamples: {N}")
        text.write("\n\nReference Frame: International Celestial Reference" + 
                   " Frame (ICRF)")
        text.write("\nReference Plane: Ecliptic X-Y Plane derived from ICRF" + 
                   " (J2000 obliquity)")
        text.write("\n\nState Vectors:\n\n")
        sigma_r_om = [sigma_decimals(sigma_r) for sigma_r in sigma_r_ec]
        sigma_v_om = [sigma_decimals(sigma_v) for sigma_v in sigma_v_ec]
        mean_rs = [f'{mean_r:.{int(om)}f}' for mean_r, om in zip(mean_r_ec, sigma_r_om)]
        sigma_rs = [f'{sigma_r:.{int(om)}f}' for sigma_r, om in zip(sigma_r_ec, sigma_r_om)]
        mean_vs = [f'{mean_v:.{int(om)}f}' for mean_v, om in zip(mean_v_ec, sigma_v_om)]
        sigma_vs = [f'{sigma_v:.{int(om)}f}' for sigma_v, om in zip(sigma_v_ec, sigma_v_om)]     
        text.write("R (AU)     = [{} {} {}] +/- [{} {} {}]\n".format(mean_rs[0],
                 mean_rs[1], mean_rs[2], sigma_rs[0], sigma_rs[1], sigma_rs[2]))
        text.write("V (AU/day) = [{} {} {}] +/- [{} {} {}]\n\n"
            .format(mean_vs[0], mean_vs[1], mean_vs[2], sigma_vs[0], 
                    sigma_vs[1], sigma_vs[2]))
        text.write("Osculating Orbital Elements:\n\n")
        sigma_coes_om = [sigma_decimals(sigma_oe) for sigma_oe in sigma_cop]
        mean_coes = [f'{mean_coe:.{int(om)}f}' for mean_coe, om in zip(mean_cop, sigma_coes_om)]
        sigma_coes = [f'{sigma_coe:.{int(om)}f}' for sigma_coe, om in zip(sigma_cop, sigma_coes_om)]
        if id in ('u', 'U'):
            text.write(f"Epoch (TDB JD) = {tdb} ({ltc_epoch.tdb.isot})\n")
            text.write(f"SBDB Match: {sbdb_name} at {sbdb_epoch} ({sbdb_epoch.iso})\n")
            text.write(f"Orbit Class: {orbit_code} - {orbit_name}\n\n")
            text.write(("{:<29}{:<27}{:<17}{:<7}\n".format("Orbital Element", 
                                                               "Calculated",
                                                               "SBDB Match", 
                                                               "Var (%)\n")))
            for i in range(len(labels)):
                if ape[i] < 1.0e-5: 
                    text.write("{:<18}{:>14} +/- {:<15}{:>13.5f}{:>15.1e}\n".format(labels[i], mean_coes[i], sigma_coes[i], sbdb_cop[i][0], ape[i]))
                else:
                    text.write("{:<18}{:>14} +/- {:<15}{:>13.5f}{:>15.5f}\n".format(labels[i], mean_coes[i], sigma_coes[i], sbdb_cop[i][0], ape[i]))
                
        else:
            text.write(f"Epoch (TDB JD) = {tdb} ({ltc_epoch.tdb.iso})\n")
            text.write(f"Orbit Class: {orbit_code} - {orbit_name}\n\n")
            text.write(("{:<29}{:<26}{:<18}{:<7}\n".format("Orbital Element",
                                                               "Calculated",
                                                               "JPL Horizons",
                                                               "APE (%)\n")))
            for i in range(len(labels)):
                if ape[i] < 1.0e-5: 
                    text.write("{:<18}{:>14} +/- {:<15}{:>13.5f}{:>15.1e}\n".format(labels[i], mean_coes[i], sigma_coes[i], jpl_cop[i][0], ape[i]))
                else:
                    text.write("{:<18}{:>14} +/- {:<15}{:>13.5f}{:>15.5f}\n".format(labels[i], mean_coes[i], sigma_coes[i], jpl_cop[i][0], ape[i]))
    print("Creating Orbit plots...")
    # Start the second plotting timer
    start_time22 = Time.now()
    # Plot the orbits
    if id in ('u', 'U'):
        theme = 'light' # Use light theme, can be changed to 'dark'
        plot_orbit(name, tdb, mean_r_ec, mean_v_ec, mean_cop, sbdb_match, theme)
    else:
        theme = 'light'
        plot_orbit(name, tdb, mean_r_ec, mean_v_ec, mean_cop, jpl_cop, theme)
    # Stop the second plotting timer and add the two
    end_time22 = Time.now()
    elapsed_time22 = end_time22 - start_time22
    elapsed_time2 = elapsed_time21 + elapsed_time22
    # Tell the user is done calculating and tell the time it took
    if elapsed_time2.sec < 60:
        print(f"All Plots created in {elapsed_time2.sec:.3f}s")
        print("All plots saved to folder.")
    elif elapsed_time2.sec < 3600:
        m, s = divmod(elapsed_time2.sec, 60)
        print(f"All Plots created in {int(m)}m {int(s)}s")
        print("All plots saved to folder.")
    else:
        h, remainder = divmod(elapsed_time2.sec, 3600)
        m, s = divmod(remainder, 60)
        print(f"All Plots created in {int(h)}h {int(m)}m {int(s)}s")
        print("All plots saved to folder.")
    # Ask user if they want to generate ephemerides
    while True:
        print("\nDo you wish to generate ephemerides for this asteroid? (YES 'Y' or NO 'N')")
        answer = input("Answer: ").upper()
        if answer in ('Y', 'YES'):
            warnings.filterwarnings("ignore", category=FutureWarning)
            # Get the ephemerides site from the user
            while True:
                print("\nEnter the desired location for the Horizons System ephemerides.")
                print("NOTE: ONLY 1 SITE is currently supported.")
                print("\n* MPC code or coordinates separated by COMMAS (lon[deg],lat[deg],alt[m]):")
                site_str = input("* ").upper()
                if re.match("^[A-Z0-9][0-9]{2}$", site_str): # Use as MPC code
                    while True:
                        try:
                            site = mpc_obs_query(site_str.upper())
                            if site.size == 0:
                                print("\nERROR: INVALID CODE. Enter a VALID MPC Observatory code.")
                                answer1 = None 
                                break
                        except Exception:
                            print("\nERROR: INVALID CODE. Enter a VALID MPC Observatory code.")
                            answer1 = None
                            break
                        print(f"\n→ {site[0, 0]} - {site[0, 6]} ({site[0, 7]})")
                        print("Is this the correct site? (YES 'Y' or NO 'N')")
                        answer1 = input("Answer: ").upper()
                        if answer1 in ('Y','YES'):
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
                elif site_str.count(',') == 2: # Use as manual location
                    coords = list(site_str.split(","))
                    while True:
                        if all(isinstance(float(coord), float) for coord in coords):
                            site = np.array(coords).reshape((1, 3))
                            answer1 = None
                        else:
                            print("ERROR: INVALID INPUT FORMAT. Enter DECIMAL values.")
                            answer1 = None
                            break
                        print(f"\n→ {site[0, 0]},{site[0, 1]} (at {site[0, 2]} m)")
                        print("Are this the correct coords? (YES 'Y' or NO 'N')")
                        answer1 = input("Answer: ").upper()
                        if answer1 in ('Y','YES'):
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
                    print("\nERROR: INVALID FORMAT. Enter in the EXACT FORMAT:")
                    print("\nMPC code or coordinates separated by COMMAS (lon[deg],lat[deg],alt[km])")
                    continue
                break
            # Get the ephemerides epochs from the user
            while True:
                print("\nEnter the desired ephemerides utc epochs.")
                print("NOTE: UP TO 180 DAYS RECOMMENDED. Further epochs could result in LARGE errors.")
                print("\n* UTC Epochs in ISOT format (yyyy-mm-ddThh:mm:ss.sss) separated by COMMAS:")
                eph_dates_str = input("* ")
                # Split the epochs string into a list and remove duplicates
                eph_dates = list(set(eph_dates_str.split(",")))
                for eph_date in eph_dates:
                    try:  # Check if the epoch is in ISO format
                        Time(eph_date, format = 'isot', scale = 'utc')
                        correct_epochs = True
                    except ValueError:
                        correct_epochs = False
                        reason = "yyyy-mm-ddThh:mm:ss.sss separated by COMMAS"
                        break
                # If the epochs are not in the correct format continue the loop
                if not correct_epochs:
                    print(
                        f"\nERROR: INVALID FORMAT. Enter the epochs in the EXACT FORMAT: {reason}")
                    # If the number of epochs is 3 and the epochs are in the correct 
                    # format, sort the epochs and break the loop 
                    continue
                if site.shape[1] == 8:
                    site_lon, site_lat = site[0, 1], site[0, 2]
                else:
                    site_lon, site_lat = site[0, 0], site[0, 1]
                eph_epchs = Time(eph_dates, format = 'isot', scale = 'utc', 
                            location = (site_lon, site_lat))
                eph_epochs = eph_epchs.sort()
                break
            # Start the ephemerides timer
            start_time3 = Time.now()
            # Calculate the mean and std of the equatorial state vectors in km
            mean_r_eq = np.array(
                [np.mean(r[:, 0]), np.mean(r[:, 1]), np.mean(r[:, 2])])
            sigma_r_eq = np.array(
                [np.std(r[:, 0]), np.std(r[:, 1]), np.std(r[:, 2])])
            mean_v_eq = np.array(
                [np.mean(v[:, 0]), np.mean(v[:, 1]), np.mean(v[:, 2])])
            sigma_v_eq = np.array(
                [np.std(v[:, 0]), np.std(v[:, 1]), np.std(v[:, 2])])
            # Generate N random state vectors values from normal distribtion 
            R0xs = np.random.normal(mean_r_eq[0], sigma_r_eq[0], N)
            R0ys = np.random.normal(mean_r_eq[1], sigma_r_eq[1], N)
            R0zs = np.random.normal(mean_r_eq[2], sigma_r_eq[2], N)
            V0xs = np.random.normal(mean_v_eq[0], sigma_v_eq[0], N)
            V0ys = np.random.normal(mean_v_eq[1], sigma_v_eq[1], N)
            V0zs = np.random.normal(mean_v_eq[2], sigma_v_eq[2], N)
            R0s = np.array([R0xs, R0ys, R0zs]).T
            V0s = np.array([V0xs, V0ys, V0zs]).T
            eph = generate_ephemerides(ltc_epoch, eph_epochs, R0s, V0s, N, ltc = False)
            eph_ltcs = eph[:, 2]/(299792.458/149597870.7)
            ltc_eph_epochs = eph_epochs - TimeDelta(eph_ltcs, format = 'sec')
            ltc_eph = generate_ephemerides(ltc_epoch, ltc_eph_epochs, R0s, V0s, N, ltc = True)
            # Get the ephemerides from JPL Horizons to comnpare
            if id in ('u', 'U'):
                sbdb_eph = jpl_horizons_ephemerides(sbdb_id, site, eph_epochs)
            else:
                jpl_eph = jpl_horizons_ephemerides(id, site, eph_epochs)         
            # Save Ephemerides to Results.txt
            with open(file, "a", encoding='utf-8') as text:
                text.write("\nAstrometric Ephemerides for the Specified Epochs:\n")
                if site.shape[1] == 8:
                    text.write(f"\nObservatory: {site[0, 0]} - {site[0, 6]} ({site[0, 7]})\n")
                else:
                    text.write(f"\nLocation: {site[0, 0]}, {site[0, 1]} (at {site[0, 2]} m)\n")
                if id in ('u', 'U'):
                    text.write("\n{:>40}{:>24}{:>16}".format("Calculated", 
                               "SBDB Match", "Var (%)"))
                    text.write("\n")
                else:
                    text.write("\n{:>40}{:>25}{:>15}".format("Calculated",
                               "JPL Horizons", "APE (%)"))
                    text.write("\n")
                for i in range(len(eph_epochs)):
                    text.write(f"\n{eph_epochs[i].iso}")
                    eph_ra_std_om = sigma_decimals(ltc_eph[i, 3])
                    eph_dec_std_om = sigma_decimals(ltc_eph[i, 4])
                    eph_ra = f'{ltc_eph[i, 0]:.{int(eph_ra_std_om)}f}'
                    eph_ra_std = f'{ltc_eph[i, 3]:.{int(eph_ra_std_om)}f}'
                    eph_dec = f'{ltc_eph[i, 1]:.{int(eph_dec_std_om)}f}'
                    eph_dec_std = f'{ltc_eph[i, 4]:.{int(eph_dec_std_om)}f}'
                    if id in ('u', 'U'):
                        sbdb_ra_std_om = sigma_decimals(sbdb_eph[i, 2])
                        sbdb_dec_std_om = sigma_decimals(sbdb_eph[i, 3])
                        sbdb_ra = f'{sbdb_eph[i, 0]:.{int(sbdb_ra_std_om)}f}'
                        sbdb_ra_std = f'{sbdb_eph[i, 2]:.{int(sbdb_ra_std_om)}f}'
                        sbdb_dec = f'{sbdb_eph[i, 1]:.{int(sbdb_dec_std_om)}f}'
                        sbdb_dec_std = f'{sbdb_eph[i, 3]:.{int(sbdb_dec_std_om)}f}'
                        text.write(
                            ("\n{:<25}{:>8} +/- {:<8}{:>11} +/- {:<11}{:<7.5f}"
                             .format("RA +/- std  [deg]", eph_ra, eph_ra_std, 
                                     sbdb_ra, sbdb_ra_std, (sbdb_eph[i, 0] - 
                                        ltc_eph[i, 0]) / sbdb_eph[i, 0] * 100)))
                        text.write(
                            ("\n{:<25}{:>8} +/- {:<8}{:>11} +/- {:<11}{:<7.5f}"
                             .format("DEC +/- std [deg]", eph_dec,
                                     eph_dec_std, sbdb_dec,
                                     sbdb_dec_std, (sbdb_eph[i, 1] - 
                                        ltc_eph[i, 1]) / sbdb_eph[i, 1] * 100)))
                    else:
                        jpl_ra_std_om = sigma_decimals(jpl_eph[i, 2])
                        jpl_dec_std_om = sigma_decimals(jpl_eph[i, 3])                        
                        jpl_ra = f'{jpl_eph[i, 0]:.{int(jpl_ra_std_om)}f}'
                        jpl_ra_std = f'{jpl_eph[i, 2]:.{int(jpl_ra_std_om)}f}'
                        jpl_dec = f'{jpl_eph[i, 1]:.{int(jpl_dec_std_om)}f}'
                        jpl_dec_std = f'{jpl_eph[i, 3]:.{int(jpl_dec_std_om)}f}'
                        text.write(
                            ("\n{:<25}{:>8} +/- {:<8}{:>11} +/- {:<11}{:<7.5f}"
                             .format("RA +/- std  [deg]", eph_ra, eph_ra_std, 
                                     jpl_ra, jpl_ra_std, abs((jpl_eph[i, 0] - 
                                        ltc_eph[i, 0]) / jpl_eph[i, 0]) * 100)))
                        text.write(
                            ("\n{:<25}{:>8} +/- {:<8}{:>11} +/- {:<11}{:<7.5f}"
                             .format("DEC +/- std [deg]", eph_dec, 
                                     eph_dec_std, jpl_dec, jpl_dec_std,
                                     abs((jpl_eph[i, 1] - ltc_eph[i, 1]) / 
                                         jpl_eph[i, 1]) * 100)))         
            # Plot the ephemerides
            print("\nCreating Ephemerides plots...")
            plt.rcParams["font.family"] = "Arial"
            if id in ('u', 'U'):
                db_eph = sbdb_eph
            else:
                db_eph = jpl_eph
            # Calculate the number of rows needed
            if len(ltc_eph) % 2 != 0:
                rows = (len(ltc_eph) + 1) // 2
            else:
                rows = len(ltc_eph) // 2
            if len(ltc_eph) == 1:
                fig, ax = plt.subplots(1, 1, figsize = (10, 10))
            elif len(ltc_eph) == 2:
                fig, ax = plt.subplots(1, 2, figsize = (12, 6))
            else:
                fig, ax = plt.subplots(rows, 2, figsize = (12, 6 * rows))
            for i in range(len(ltc_eph)):
                if len(ltc_eph) == 1:
                    ax_i = ax
                elif len(ltc_eph) == 2:
                    ax_i = ax[i]
                else:
                    row = i // 2
                    col = i % 2
                    ax_i = ax[row, col]
                ax_i.errorbar(ltc_eph[i, 0], ltc_eph[i, 1], xerr = ltc_eph[i, 3], yerr = ltc_eph[i, 4],
                                label = 'Calculated', marker = 'o', c = 'r', alpha = 1,
                                zorder = 1)
                ax_i.errorbar(db_eph[i, 0], db_eph[i, 1], xerr = db_eph[i, 2], yerr = db_eph[i, 3],
                                label = 'JPL Horizons', marker = 's', c = 'g', alpha = 1,
                                zorder = 2)
                ax_i.margins(x = 0.5, y = 0.5)
                ax_i.set_xlabel('α [deg]', fontweight = 'bold', fontsize = 12)
                ax_i.set_ylabel('δ [deg]', fontweight = 'bold', fontsize = 12)                
                ax_i.set_title(f"Epoch: {eph_epochs[i].iso}", 
                                fontweight = 'bold', fontsize = 14)
                ax_i.legend(fontsize = 12)
                ax_i.ticklabel_format(useOffset = False, style = 'plain')
                ax_i.xaxis.set_major_locator(plt.MaxNLocator(nbins = 5))
                ax_i.yaxis.set_major_locator(plt.MaxNLocator(nbins = 5))
                ax_i.grid(c = 'gray', ls = 'dashed', alpha = 0.25)
                ax_i.set_axisbelow(True)
            # Remove empty subplots if necessary
            if len(ltc_eph) > 2 and len(ltc_eph) % 2 != 0:
                fig.delaxes(ax[rows - 1, 1])
            fig.suptitle(f'Astrometric Ephemerides of {name}', fontsize = 18, fontweight = 'bold')
            plt.tight_layout(rect = [0, 0, 1, 0.99])
            plt.savefig(ephemerides, format = "png", dpi = 300) # Save the plots
            plt.close('all')
            # Stop the ephemerides timer
            end_time3 = Time.now()
            elapsed_time3 = end_time3 - start_time3
            # Tell the user is done calculating and tell the time it took
            if elapsed_time3.sec < 60:
                print(f"\nEphemerides calculated in {elapsed_time3.sec:.3f}s")
                print("Ephemerides added to Orbit file.")
                print("Plots saved to folder.")
            elif elapsed_time3.sec < 3600:
                m, s = divmod(elapsed_time3.sec, 60)
                print(f"\nEphemerides calculated in {int(m)}m {s:.3f}s")
                print("Ephemerides added to Orbit file.")
                print("Plots saved to folder.")
            else:
                h, remainder = divmod(elapsed_time3.sec, 3600)
                m, s = divmod(remainder, 60)
                print(
                    f"\nEphemerides calculated in {int(h)}h {int(m)}m {s:.3f}s")
                print("Ephemerides added to Orbit file.")
                print("Plots saved to folder.")
            elapsed_time = elapsed_time1 + elapsed_time2 + elapsed_time3
            with open(file, "a", encoding='utf-8') as text:
                # Tell the user and save the total calculating time
                # and end the orbit calculation
                if elapsed_time.sec < 60:
                    text.write(
                        f"\n\nTotal Calculation Time: {elapsed_time.sec:.3f}s")
                    print(f"\nTotal Calculation Time: {elapsed_time.sec:.3f}s")
                elif elapsed_time.sec < 3600:
                    m, s = divmod(elapsed_time.sec, 60)
                    text.write(f"\n\nTotal Calculation Time: {int(m)}m {s:.3f}s")
                    print(f"\nTotal Calculation Time: {int(m)}m {s:.3f}s")
                else:
                    h, remainder = divmod(elapsed_time.sec, 3600)
                    m, s = divmod(remainder, 60)
                    text.write(
                        f"\n\nTotal Calculation Time: {int(h)}h {int(m)}m {s:.3f}s")
                    print(
                        f"\nTotal Calculation Time: {int(h)}h {int(m)}m {s:.3f}s")
                text.write("\n\n{:-^80}".format("# END #"))
            print(f"\nResults saved to folder in file '{name} - Results.txt'.")
            break
        elif answer in ('N', 'NO'):
            elapsed_time = elapsed_time1 + elapsed_time2
            with open(file, "a", encoding='utf-8') as text:
                # Tell the user and save the total calculating time
                # and end the orbit calculation
                if elapsed_time.sec < 60:
                    text.write(
                        f"\nTotal Calculation Time: {elapsed_time.sec:.3f}s")
                    print(f"\nTotal Calculation Time: {elapsed_time.sec:.3f}s")
                elif elapsed_time.sec < 3600:
                    m, s = divmod(elapsed_time.sec, 60)
                    text.write(f"\nTotal Calculation Time: {int(m)}m {s:.3f}s")
                    print(f"\nTotal Calculation Time: {int(m)}m {s:.3f}s")
                else:
                    h, remainder = divmod(elapsed_time.sec, 3600)
                    m, s = divmod(remainder, 60)
                    text.write(
                        f"\nTotal Calculation Time: {int(h)}h {int(m)}m {s:.3f}s")
                    print(
                        f"\nTotal Calculation Time: {int(h)}h {int(m)}m {s:.3f}s")
                text.write("\n\n{:-^80}".format("# END #"))
            print(f"\nResults saved to folder in file '{name} - Results.txt'.")
            break
        else:
            print("ERROR: Answer must be YES (Y) or NO (N). Enter a valid answer.")
    print("\n{:-^80}".format(" Orbit Done "))

if __name__ == "__main__":
    # Program in a loop in case the user wants to calculate multiple orbits
    while True:
        # Check the internet connection
        connection = check_internet_connection()
        if connection is True:
            initial_orbit_determination() # Run Program
            # Ask the user whether they want to calculate a new orbit or quit
            while True:
                print("\nDo you want to calculate another orbit? (YES 'Y' or NO 'N')")
                reset = input("Answer: ").upper()
                if reset in ('N','NO'):
                    break
                elif reset in ('Y','YES'):
                    print()
                    break
                else:
                    print("ERROR: Answer must be YES (Y) or NO (N). Enter a valid answer.")
            if reset in ('N','NO'):
                print("\n{:~^80}".format("# End of Program #"))
                break
            else:
                continue
        else:
            # Ask the user whether they want to retry connection or quit
            print("WARNING: NO INTERNET CONNECTION.")
            print("This program relies on APIs, so an internet connection IS REQUIERED.")
            print("\nDo you want to try again? (YES 'Y' or NO 'N')")
            retry = input("Answer: ").upper()
            if retry in ('N','NO'):
                print("\n{:~^80}".format("# End of Program #"))
                break
            elif retry in ('Y','YES'):
                continue
            else:
                print("ERROR: Answer must be YES (Y) or NO (N). Enter a valid answer.")
