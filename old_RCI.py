# Revised Copenhagen Interpretation validation script.
# dependencies.
import os
import datetime
import asyncio
import multiprocessing
import threading
import numpy as np
import scipy
import cupy
import pandas as pd
import torch
import matplotlib
import logging
# Logging set up
logging.basicConfig(level=logging.DEBUG)
# universal constants dictionary.
constants_dict = {
    'c': 299792458,     # Speed of Light.
    'G': 6.67430e-11,    # Gravitational constant.
    'h': 6.62607015e-34,    # Planck's constant.
    'e': 1.602176634e-19,    # Elementary charge.
    'K_B': 1.380649e-23,    # Boltzmann Constant.
    'Pi': np.pi,    # Pi obviously.
    'lp': 1.616255e-35,    # Planck Length.
    'mp': 2.176434e-8,    # Planck Mass.
    'Tp': 1.416784e32,    # Plank Temperature.
    'tp': 5.391274e-44,    # Plank time in seconds,
    'h_bar': 1.054571817e-34,   # Reduced Planck's constant.
    'e_0': 8.8541878188e-12,    # Electric permissivity of freespace.
    'u_0': 1.25663706127e-6,    # Magnetic permissivity of freespace.
    'Z_0': 376.730313412,    # Characteristic impedance of freespace.
    'omega': 5.670374419e-8,    # Stefan-Boltzmann constant.
    'nM': 1.67492750056e-27,    # Neutron mass.
    'pM': 1.67262192595e-27,    # Proton mass.
}
# Observational data. Awaiting potential expansion and revision.
observational_data = {
    'H_0_CMB': 67.74,    # Hubble constant as measured through CMB observations.
    'H_0_local': 73.2,    # Hubble constant as measure through local observations (Type 1a supernovae).
    'CMB_E_field': 1e-10,   # Estimation provided by chat GPT.
    'CMB_B_field': 1e-12,   # Estimation provided by chat GPT.
    'OU_E_field': 1e-12,   # Estimation provided by chat GPT.
    'OU_B_field': 1e-15,   # Estimation provided by chat GPT.
    'WMAP_redshift': 1089
}
assumptions_data = {
    'Omega_matter': 0.3,    # Approximate Universal matter density.
    'Omega_lambda': 0.7,    # Approximate Universal dark matter density.
    'OU_age': 4.351968e17,    # Approximate age of the Observable Universe in seconds
    'Time_epoch': 1.198368e13    # Approximate age of the universe at CMB emission.
}
# Dictionary creation.
constants_dict = dict(constants_dict)
observational_data = dict(observational_data)
assumptions_data = dict(assumptions_data)
# Region scale data. Awaiting refinement
Regions_data = {
    'Region': ['CMB', 'MW', 'OU'],   # Names of regions used for scaling.
    'Mass': [1e-18, 1e12, 9e53],   # Mass contained within each region/scale.
    'Temperature': [3000, 2.725, 2.725],   # Temperature of the region/scale in Kelvin.
    'Volume_diameter': [7.56858438e+23, 1.892146095e+24, 8.799993e17],    # Diameter of the volume of the region/scale in meters.
}
# Relavent atomic data. Awaiting expansion.
Atomic_data = {
    'Atom': ['H', 'H2', 'H3', 'He'],
    'Atomic_N': [1, 1, 1, 2,],
    'Neutron_count': [0, 1, 2, 1],
    'Atomic_M': [1.007825031898, 2.014101777844, 3.01604928132, 3.01602932197],
}
# Data frame creation.
regions_df = pd.DataFrame(Regions_data)
atoms_df = pd.DataFrame(Atomic_data)
# Rho_E_x volume calculation for CMB.
async def calculate_CMB_volume():
    # Constants.
    Pi = constants_dict['Pi']
    # Region data.
    Volume_diameter = regions_df.set_index('Region').at['CMB', 'Volume_diameter']
    # Volume calculation.
    V = (4/3)* Pi * (Volume_diameter ** 3)
    # Debug logging.
    logging.debug(f"Calculated CMB Volume: {V}")
    # Return
    return V
# Rho_E_x volume calculation for Milkyway Galaxy.
async def calculate_MW_volume():
    # Constants.
    Pi = constants_dict['Pi']
    # Region data.
    Volume_diameter = regions_df.set_index('Region').at['MW', 'Volume_diameter']
    # Volume calculation.
    V = (4/3)* Pi * (Volume_diameter ** 3)
    # Debug logging.
    logging.debug(f"Calculated MW Volume: {V}")
    # Return.
    return V
# Rho_E volume calculation for the Observable Universe.
async def calculate_OU_volume():
    # Constants.
    Pi = constants_dict['Pi']
    # Region data.
    Volume_diameter = regions_df.set_index('Region').at['OU', 'Volume_diameter']
    # Volume calculation.
    V = (4/3)* Pi * (Volume_diameter ** 3)
    # Debug logging.
    logging.debug(f"Calculated OU Volume: {V}")
    # Return.
    return V
# Early Universe Surface Area Calculation
async def Calculate_CMB_surface_area():
    # Constants.
    Pi = constants_dict['Pi']
    # Region data.
    r2 = regions_df.set_index('Region').at['CMB', 'Volume_diameter']
    # Area calculation.
    A = 4*Pi*r2
    # Debug logging.
    logging.debug(f"Calculated CMB Surface Area: {A}")
    # Return.
    return A
# Milkyway Galaxy Surface Area Calculation
async def Calculate_MW_surface_area():
    # Constants.
    Pi = constants_dict['Pi']
    # Region data.
    r2 = regions_df.set_index('Region').at['MW', 'Volume_diameter']
    # Area calculation.
    A = 4*Pi*r2
    # Debug logging.
    logging.debug(f"Calculated MW Surface Area: {A}")
    # Return.
    return A
# Observable Universe Galaxy Surface Area Calculation
async def Calculate_OU_surface_area():
    # Constants.
    Pi = constants_dict['Pi']
    # Region data.
    r2 = regions_df.set_index('Region').at['OU', 'Volume_diameter']
    # Area calculation.
    A = 4*Pi*r2
    # Debug logging.
    logging.debug(f"Calculated OU Surface Area: {A}")
    # Return.
    return A
# Early universe Mass Energy conversion.
async def calculate_CMB_Mass_energy():
    # Constants.
    c = constants_dict['c']
    # Region data.
    M = regions_df.set_index('Region').at['CMB', 'Mass']
    # Mass energy calculation.
    E_Mass = M * (c ** 2)
    # Debug logging.
    logging.debug(f"Calculated CMB Mass Energy: {E_Mass}")
    # Return.
    return E_Mass
# Milkyway Mass Energy conversion.
async def calculate_MW_Mass_energy():
    # Constants.
    c = constants_dict['c']
    # Region data.
    M = regions_df.set_index('Region').at['MW', 'Mass']
    # Mass energy calculation.
    E_Mass = M * (c ** 2)
    # Debug logging.
    logging.debug(f"Calculated MW Mass Energy: {E_Mass}")
    # Return.
    return E_Mass
# Observable Universe Mass Energy conversion.
async def calculate_OU_Mass_energy():
    # Constants.
    c = constants_dict['c']
    # Region data.
    M = regions_df.set_index('Region').at['OU', 'Mass']
    # Mass energy calculation.
    E_Mass = M * (c ** 2)
    # Debug logging.
    logging.debug(f"Calculated OU Mass Energy: {E_Mass}")
    # Return.
    return E_Mass
# Early Universe Temperature Energy conversion.
async def calculate_CMB_Temp_energy():
    # Constants.
    K_B = constants_dict['K_B']
    # Region data.
    T = regions_df.set_index('Region').at['CMB', 'Temperature']
    E_Temp = K_B * T
    logging.debug(f"Calculated CMB Temperature energy : {E_Temp}")
    # Return.
    return E_Temp
# Milkyway Temperature Energy conversion.
async def calculate_MW_Temp_energy():
    # Constants.
    K_B = constants_dict['K_B']
    # Region data.
    T = regions_df.set_index('Region').at['MW', 'Temperature']
    E_Temp = K_B * T
    logging.debug(f"Calculated MW Temperature energy : {E_Temp}")
    # Return.
    return E_Temp
# Observable Universe Temperature Energy conversion.
async def calculate_OU_Temp_energy():
    # Constants.
    K_B = constants_dict['K_B']
    # Region data.
    T = regions_df.set_index('Region').at['OU', 'Temperature']
    E_Temp = K_B * T
    logging.debug(f"Calculated OU Temperature energy : {E_Temp}")
    # Return.
    return E_Temp
# Early Universe Electric energy calculation.
async def calculate_CMB_U_E_energy():
    # Constants.
    e_0 = constants_dict['e_0']
    # Observational data.
    E = observational_data['CMB_E_field']
    # Calculation dependency.
    V = await calculate_CMB_volume()
    # Electric Field energy calculation
    U_E = 0.5 * e_0 * (E ** 2) * V
    # debug logging.
    logging.debug(f"Calculated CMB Electric Field Energy: {U_E}")
    # Return.
    return U_E
# Milkyway Electric energy calculation.
async def calculate_MW_U_E_energy():
    # Constants.
    e_0 = constants_dict['e_0']
    # Observational data.
    E = observational_data['MW_E_field']
    # Calculation dependency.
    V = await calculate_MW_volume()
    # Electric Field energy calculation
    U_E = 0.5 * e_0 * (E ** 2) * V
    # debug logging.
    logging.debug(f"Calculated MW Electric Field Energy: {U_E}")
    # Return.
    return U_E
# Observable Universe Electric energy calculation.
async def calculate_OU_U_E_energy():
    # Constants.
    e_0 = constants_dict['e_0']
    # Observational data.
    E = observational_data['OU_E_field']
    # Calculation dependency.
    V = await calculate_OU_volume()
    # Electric Field energy calculation
    U_E = 0.5 * e_0 * (E ** 2) * V
    # debug logging.
    logging.debug(f"Calculated OU Electric Field Energy: {U_E}")
    # Return.
    return U_E
# Early universe Magnetic energy calculation.
async def calculate_CMB_U_B_energy():
    # Constants.
    u_0 = constants_dict['u_0']
    # Observational data.
    B = observational_data['CMB_B_field']
    # Calculation dependency.
    V = await calculate_CMB_volume()
    # Magnetic field energy calculation.
    U_B = (1/(2 * u_0)) * (B **2) * V
    # Debug logging.
    logging.debug(f"Calculated CMB Magnetic Field Energy: {U_B}")
    # Return.
    return U_B
# Milkyway Magnetic energy calculation.
async def calculate_MW_U_B_energy():
    # Constants.
    u_0 = constants_dict['u_0']
    # Observational data.
    B = observational_data['MW_B_field']
    # Calculation dependency.
    V = await calculate_MW_volume()
    # Magnetic field energy calculation.
    U_B = (1/(2 * u_0)) * (B **2) * V
    # Debug logging.
    logging.debug(f"Calculated MW Magnetic Field Energy: {U_B}")
    # Return.
    return U_B
# Observable Universe Magnetic energy calculation.
async def calculate_OU_U_B_energy():
    # Constants.
    u_0 = constants_dict['u_0']
    # Observational data.
    B = observational_data['OU_B_field']
    # Calculation dependency.
    V = await calculate_OU_volume()
    # Magnetic field energy calculation.
    U_B = (1/(2 * u_0)) * (B **2) * V
    # Debug logging.
    logging.debug(f"Calculated OU Magnetic Field Energy: {U_B}")
    # Return.
    return U_B
# Early Universe Nuclear binding energy calculation.
async def calculate_H_Nuclear_binding_energy():
    # Constants:
    c = constants_dict['c']
    nM = constants_dict['nM']
    pM = constants_dict['pM']
    # Atomic Data:
    Z = atoms_df.set_index('Atom').at['H', 'Atomic_N']
    N = atoms_df.set_index('Atom').at['H', 'Neutron_count']
    M = atoms_df.set_index('Atom').at['H', 'Atomic_M']
    Nucleus_M = (M * 1.66053906660e-27)
    # Mass Defect Calculation.
    mass_defect = (Z * pM + N * nM) - Nucleus_M
    # Nuclear Binding Energy Calculation.
    E_Nuclear = mass_defect * (c ** 2)
    # Debug logging
    logging.debug(f"Calculated Hydrogen nuclear binding energy: {E_Nuclear}")
    # Return
    return E_Nuclear
# Early Universe Nuclear binding energy calculation.
async def calculate_H2_Nuclear_binding_energy():
    # Constants:
    c = constants_dict['c']
    nM = constants_dict['nM']
    pM = constants_dict['pM']
    # Atomic Data:
    Z = atoms_df.set_index('Atom').at['H2', 'Atomic_N']
    N = atoms_df.set_index('Atom').at['H2', 'Neutron_count']
    M = atoms_df.set_index('Atom').at['H2', 'Atomic_M']
    Nucleus_M = (M * 1.66053906660e-27)
    # Mass Defect Calculation.
    mass_defect = (Z * pM + N * nM) - Nucleus_M
    # Nuclear Binding Energy Calculation.
    E_Nuclear = mass_defect * (c ** 2)
    # Debug logging.
    logging.debug(f"Calculated Deuterium nuclear binding energy: {E_Nuclear}")
    # Return
    return E_Nuclear
# Early Universe Nuclear binding energy calculation.
async def calculate_H3_Nuclear_binding_energy():
    # Constants:
    c = constants_dict['c']
    nM = constants_dict['nM']
    pM = constants_dict['pM']
    # Atomic Data:
    Z = atoms_df.set_index('Atom').at['H3', 'Atomic_N']
    N = atoms_df.set_index('Atom').at['H3', 'Neutron_count']
    M = atoms_df.set_index('Atom').at['H3', 'Atomic_M']
    Nucleus_M = (M * 1.66053906660e-27)
    # Mass Defect Calculation.
    mass_defect = (Z * pM + N * nM) - Nucleus_M
    # Nuclear Binding Energy Calculation.
    E_Nuclear = mass_defect * (c ** 2)
    # Debug logging.
    logging.debug(f"Calculated Tritium nuclear binding energy: {E_Nuclear}")
    # Return
    return E_Nuclear
# Early Universe Nuclear binding energy calculation.
async def calculate_He_Nuclear_binding_energy():
    # Constants:
    c = constants_dict['c']
    nM = constants_dict['nM']
    pM = constants_dict['pM']
    # Atomic Data:
    Z = atoms_df.set_index('Atom').at['He', 'Atomic_N']
    N = atoms_df.set_index('Atom').at['He', 'Neutron_count']
    M = atoms_df.set_index('Atom').at['He', 'Atomic_M']
    Nucleus_M = (M * 1.66053906660e-27)
    # Mass Defect Calculation.
    mass_defect = (Z * pM + N * nM) - Nucleus_M
    # Nuclear Binding Energy Calculation.
    E_Nuclear = mass_defect * (c ** 2)
    # Debug logging.
    logging.debug(f"Calculated Helium nuclear binding energy : {E_Nuclear}")
    # Return
    return E_Nuclear
# Milkyway Elastic energy calculation.
    # E_elastic=  1/(32π(6.7430*10^(-11 ) m^3 kg^(-1) s^(-2))) ⟨h ̇_ij ┤ ├ h ̇^ij ⟩
# Early Universe H_0 energy converstion calculation.
async def H_0_CMB_energy_conversion():
    # Region data.
    H_0_CMB = observational_data['H_0_CMB']
    H_0_meters = H_0_CMB*1e3
    Mpc_meters = 3.086*1e22
    E_H_0_CMB = H_0_meters / Mpc_meters
    # Debug logging.
    logging.debug(f"Calculated CMB H_0 energy conversion: {E_H_0_CMB}")
    # Return.
    return E_H_0_CMB
# Local_H_0 energy conversion calculation.
async def H_0_local_energy_conversion():
    # Region data.
    H_0_local = observational_data['H_0_local']
    H_0_meters = H_0_local*1e3
    Mpc_meters = 3.086*1e22
    E_H_0_local = H_0_meters / Mpc_meters
    # Debug logging.
    logging.debug(f"Calculated CMB H_0 energy conversion: {E_H_0_local}")
    # Return.
    return E_H_0_local
# Delta_H_0 calculation.
async def Calculate_Delta_H_0():
    # Region data.
    H_0_CMB = observational_data['H_0_CMB']
    H_0_local = observational_data['H_0_local']
    Delta_H_0 = H_0_local - H_0_CMB
    # Return.
    return Delta_H_0
# Delta_H_0 Energy calculation.
async def Delta_H_0_energy_conversion():
    E_H_0_CMB = await H_0_CMB_energy_conversion()
    E_H_0_local = await H_0_local_energy_conversion()
    E_Delta_H_0 = E_H_0_local - E_H_0_CMB
    # Return.
    return E_Delta_H_0
# t_hubble calculation.
async def calculate_CMB_time():
    H_0_CMB = await H_0_CMB_energy_conversion()
    t_hubble = 1/H_0_CMB
    # Debug logging
    logging.debug(f"Calculated CMB hubble time: {t_hubble}")
    # Return.
    return t_hubble
# Early Universe total Photonic energy calculation.
async def calculate_CMB_Q_E_photon_energy():
    # Constants.
    h = constants_dict['h']
    c = constants_dict['c']
    # Region data.
    l = regions_df.set_index('Region').at['CMB', 'Temperature']
    # Calculation dependencies.
    t_hubble = await calculate_CMB_time()
    # Photonic energy calculation.
    Q_E_photon = ((h * c) / l) * t_hubble
    # Debug logging
    logging.debug(f"Calculated CMB total photonic energy: {Q_E_photon}")
    # Return.
    return Q_E_photon
# Milyway total Photonic energy calculation.
async def calculate_local_Q_E_photon_energy():
    # Constants.
    h = constants_dict['h']
    c = constants_dict['c']
    # Region data.
    l = regions_df.set_index('Region').at['MW', 'Temperature']
    # Calculation dependencies.
    t_hubble = await calculate_CMB_time()
    # Photonic energy calculation.
    Q_E_photon = ((h * c) / l) * t_hubble
    # Debug logging
    logging.debug(f"Calculated MW total photonic energy: {Q_E_photon}")
    # Return.
    return Q_E_photon
# Observable Universe total Photonic energy calculation.
async def calculate_OU_Q_E_photon_energy():
    # Constants.
    h = constants_dict['h']
    c = constants_dict['c']
    # Region data.
    l = regions_df.set_index('Region').at['OU', 'Temperature']
    # Calculation dependencies.
    t_hubble = await calculate_CMB_time()
    # Photonic energy calculation.
    Q_E_photon = ((h * c) / l) * t_hubble
    # Debug logging.
    logging.debug(f"Calculated MW total photonic energy: {Q_E_photon}")
    # Return.
    return Q_E_photon
# Early universe Total Radiative energy calculation.
async def calculate_CMB_Radiative_energy():
    # Constants.
    omega = constants_dict['omega']
    # Region data.
    T = regions_df.set_index('Region').at['CMB', 'Temperature']
    A = await Calculate_CMB_surface_area()
    t_hubble = await calculate_CMB_time()
    Q_P = (omega * A * T ** 4) * t_hubble
    # Debug Logging.
    logging.debug(f"Calculated CMB Radiative energy: {Q_P}")
    # Return.
    return Q_P
# Milkyway Total Radiative energy calculation.
async def calculate_MW_Radiative_energy():
    # Constants.
    omega = constants_dict['omega']
    # Region data.
    T = regions_df.set_index('Region').at['MW', 'Temperature']
    # Calculation dependencies.
    A = await Calculate_MW_surface_area()
    t_hubble = await calculate_CMB_time()
    # Radiative energy calculation.
    Q_P = (omega * A * T ** 4) * t_hubble
    # Debug Logging.
    logging.debug(f"Calculated MW Radiative energy: {Q_P}")
    # Return.
    return Q_P
# Observable Universe Total Radiative energy calculation.
async def calculate_OU_Radiative_energy():
    # Constants.
    omega = constants_dict['omega']
    # Region data.
    T = regions_df.set_index('Region').at['OU', 'Temperature']
    # Calculation dependencies.
    A = await Calculate_OU_surface_area()
    t_hubble = await calculate_CMB_time()
    # Radiative energy calculation.
    Q_P = (omega * A * T ** 4) * t_hubble
    # Debug Logging.
    logging.debug(f"Calculated OU Radiative energy: {Q_P}")
    # Return.
    return Q_P
# Rho_E_x calculation for CMB emission.
async def calculate_CMB_Rho_E_x():
    # Calculation dependencies:
    E_mass = await calculate_CMB_Mass_energy()
    E_temp = await calculate_CMB_Temp_energy()
    U_E = await calculate_CMB_U_E_energy()
    U_B = await calculate_CMB_U_B_energy()
    E_nuclear = await calculate_H_Nuclear_binding_energy() + await calculate_H2_Nuclear_binding_energy() + await calculate_H3_Nuclear_binding_energy()
    Q_E_photon = await calculate_CMB_Q_E_photon_energy()
    QP = await calculate_CMB_Radiative_energy()
    E_H_0_CMB = await H_0_CMB_energy_conversion()
    V = await calculate_CMB_volume()
    t_hubble = await calculate_CMB_time()
    E_total = (E_mass * E_nuclear) + E_temp + U_E + U_B + Q_E_photon + QP + E_H_0_CMB
    # CMB Rho_E (x) calculation.
    CMB_Rho_E_x = (E_total / V) * t_hubble
    # Debug logging.
    logging.debug(f"Calculated CMB Rho_E (x) : {CMB_Rho_E_x}")
    # Return
    return CMB_Rho_E_x
# Scale factor calculation.
async def calculate_scale_factor():
    # Assumptions
    OU_age = assumptions_data['OU_age']
    CMB_epoch = assumptions_data['Time_epoch']
    # Redshift calcualtion.
    Z = (OU_age / CMB_epoch) - 1
    # Scaling factor calculation.
    a_t = 1 / (1 + Z)
    # Debug logging
    logging.debug(f"Caclulated redshift factor Z: {Z}")
    logging.debug(f"Caclulated scaling factor a(t): {a_t}")
    # Return
    return a_t
# Lambda Calculation.
async def calculate_Lambda():
    # Region data.
    H_0_CMB = observational_data['H_0_CMB']
    # Calculation Dependencies.
    Rho_E_x = await calculate_CMB_Rho_E_x()
    t_hubble = await calculate_CMB_time()
    a_t = await calculate_scale_factor()
    # Lambda calculation.
    L = (H_0_CMB * (Rho_E_x / t_hubble))/a_t
    # Debug Logging.
    logging.debug(f"Calculated Lambda : {L}")
    # Return.
    return L
# Standard model Delta calculations.
async def standard_model_comparison():
    # Constants:
    c = constants_dict['c']
    G = constants_dict['G']
    Pi = constants_dict['Pi']
    omega = constants_dict['omega']
    # Assumptions:
    Omega_matter = assumptions_data['Omega_matter']
    Omega_lambda = assumptions_data['Omega_lambda']
    # Region data:
    T = regions_df.set_index('Region').at['CMB', 'Temperature']
    # Caclulation dependencies:
    H_0_CMB = await H_0_CMB_energy_conversion()
    CMB_Rho_E_x = await calculate_CMB_Rho_E_x()
    # Critical Energy density calculation.
    Rho_Critical = (3 * H_0_CMB ** 2) / (8 * Pi * G)
    # Standard energy density calculations.
    Rho_matter_standard = Omega_matter * Rho_Critical
    Rho_Lambda_standard = Omega_lambda * Rho_Critical
    # Rho_rad calculation using Stefan-Boltzmann constant.
    Rho_rad = (4 * omega / c) * (T ** 4)
    # Delta Rho_E (x) calculations.
    Delta_Rho_E_x = np.abs(CMB_Rho_E_x - Rho_rad)
    # Standard model Rho calculation.
    Rho_total_standard = Rho_rad + Rho_matter_standard + Rho_Lambda_standard
    # Debug logging.
    logging.debug(f"Standard Matter Energy Density (Rho_matter_standard): {Rho_matter_standard}")
    logging.debug(f"Standard Dark Energy Density (Rho_lambda_standard): {Rho_Lambda_standard}")
    logging.debug(f"Calculated radiation energy density (Rho_rad): {Rho_rad}")
    logging.debug(f"Calculated Delta Rho_E (x) : {Delta_Rho_E_x}")
    logging.debug(f"Total Standard Energy Density (Rho_total_standard): {Rho_total_standard}")
    # Return
    return Rho_rad, Delta_Rho_E_x, Rho_total_standard
# Current Rho_E of the observable universe calculation.
async def calculate_universal_Rho_E():
    # Calculation dependencies:
    E_mass = await calculate_OU_Mass_energy()
    E_temp = await calculate_OU_Temp_energy()
    U_E = await calculate_OU_U_E_energy()
    U_B = await calculate_OU_U_B_energy()
    E_nuclear = await calculate_H_Nuclear_binding_energy() + await calculate_H2_Nuclear_binding_energy() + await calculate_H3_Nuclear_binding_energy() + await calculate_He_Nuclear_binding_energy()
    Q_E_photon = await calculate_OU_Q_E_photon_energy()
    QP = await calculate_OU_Radiative_energy()
    E_H_0_CMB = await H_0_CMB_energy_conversion()
    E_Delta_H_0 = await Delta_H_0_energy_conversion()
    V = await calculate_OU_volume()
    t_hubble = await calculate_CMB_time()
    L = await calculate_Lambda()
    E_total = (E_mass * E_nuclear) + E_temp + U_E + U_B + Q_E_photon + QP + E_H_0_CMB + E_Delta_H_0
    # CMB Rho_E (x) calculation.
    Rho_E = ((E_total / V) * t_hubble)* L
    # Debug logging.
    logging.debug(f"Calculated Rho_E: {Rho_E}")
    # Return
    return Rho_E
# Main script.
async def main():
    # Awaiting calculations.
    CMB_Rho_E_x = await calculate_CMB_Rho_E_x()
    Lambda = await calculate_Lambda()
    Rho_rad, Delta_Rho_E_x, Rho_total_standard = await standard_model_comparison()
    Rho_E = await calculate_universal_Rho_E()
    # Comparison metrics.
    CMB_Rho_E_x_percentage_difference = (Delta_Rho_E_x / Rho_rad) * 100
    CMB_Rho_E_x_ratio = CMB_Rho_E_x / Rho_rad
    Delta_Rho_E = np.abs(Rho_E - Rho_total_standard)
    Rho_E_percentage_difference = (Delta_Rho_E / Rho_total_standard) * 100
    Rho_E_ratio = Rho_E / Rho_total_standard
    # Prining explicit results.
    print(f"Calculated Rho_E (x) of CMB: {CMB_Rho_E_x}")
    print(f"Calculated value of Lambda: {Lambda}")
    print(f"Calculated radiation energy density (Rho_rad): {Rho_rad}")
    print(f"Calculated Delta Rho_E (x): {Delta_Rho_E_x}")
    print(f"Calculated Rho_E (x) Percentage difference: {CMB_Rho_E_x_percentage_difference}")
    print(f"Calculated Rho_E (x) Ratio: {CMB_Rho_E_x_ratio}")
    print(f"Calculated Universal Rho_E: {Rho_E}")
    print(f"Calculated standard Model Energy density: {Rho_total_standard}")
    print(f"Rho_E percentage difference: {Rho_E_percentage_difference}")
    print(f"Caculated Rho_E ratio: {Rho_E_ratio}")
# Running main in events loop.
if __name__ == "__main__":
    asyncio.run(main())