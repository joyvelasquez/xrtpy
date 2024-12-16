from datetime import datetime
from pathlib import Path

import pytest
from astropy import units as u

from xrtpy.response.channel import Channel
from xrtpy.response.effective_area import EffectiveAreaFundamental

channel_names = [
    "Al-mesh",
    "Al-poly",
    "C-poly",
    "Ti-poly",
    "Be-thin",
    "Be-med",
    "Al-med",
    "Al-thick",
    "Be-thick",
    "Al-poly/Al-mesh",
    "Al-poly/Ti-poly",
    "Al-poly/Al-thick",
    "Al-poly/Be-thick",
    "C-poly/Ti-poly",
]

channel_single_filter_names = [
    "Al-mesh",
    "Al-poly",
    "C-poly",
    "Ti-poly",
    "Be-thin",
    "Be-med",
    "Al-med",
    "Al-thick",
    "Be-thick",
]

valid_dates = [
    datetime(year=2006, month=9, day=25, hour=22, minute=1, second=1),
    datetime(year=2007, month=9, day=22, hour=22, minute=1, second=1),
    datetime(year=2009, month=9, day=22, hour=22, minute=1, second=1),
    datetime(year=2010, month=9, day=22, hour=22, minute=1, second=1),
    datetime(year=2012, month=9, day=22, hour=22, minute=1, second=1),
    datetime(year=2015, month=9, day=22, hour=22, minute=1, second=1),
    datetime(year=2017, month=9, day=22, hour=22, minute=1, second=1),
    datetime(year=2019, month=9, day=23, hour=22, minute=1, second=1),
    datetime(year=2020, month=9, day=23, hour=22, minute=1, second=1),
    datetime(year=2021, month=9, day=23, hour=22, minute=1, second=1),
    datetime(year=2022, month=9, day=23, hour=22, minute=1, second=1),
]

invalid_dates = [
    datetime(year=2006, month=8, day=25, hour=22, minute=1, second=1),
    datetime(year=2005, month=9, day=22, hour=22, minute=1, second=1),
    datetime(year=2002, month=9, day=22, hour=22, minute=1, second=1),
    datetime(year=2000, month=9, day=22, hour=22, minute=1, second=1),
    datetime(year=1990, month=9, day=22, hour=22, minute=1, second=1),
]


@pytest.mark.parametrize("channel_name", channel_names)
def test_channel_name(channel_name):
    channel = Channel(channel_name)
    assert channel.name == channel_name


@pytest.mark.parametrize("name", channel_names)
def test_EffectiveArea_filter_name(name):
    instance = EffectiveAreaFundamental(
        name, datetime(year=2013, month=9, day=22, hour=22, minute=0, second=0)
    )
    actual_attr_value = instance.name

    assert actual_attr_value == name


@pytest.mark.parametrize("date", valid_dates)
@pytest.mark.parametrize("name", channel_names)
def test_EffectiveArea_contamination_on_CCD(name, date):
    instance = EffectiveAreaFundamental(name, date)
    assert 0 <= instance.contamination_on_CCD <= 1206


@pytest.mark.parametrize("date", valid_dates)
@pytest.mark.parametrize("name", channel_single_filter_names)
def test_EffectiveArea_contamination_on_filter(name, date):
    instance = EffectiveAreaFundamental(name, date)
    assert 0 <= instance.contamination_on_filter <= 2901


@pytest.mark.parametrize("date", invalid_dates)
@pytest.mark.parametrize("name", channel_names)
def test_EffectiveArea_exception_is_raised(name, date):
    with pytest.raises(ValueError):  # noqa: PT011
        EffectiveAreaFundamental(name, date)


# def get_IDL_data_files():
#     directory = (
#         Path(__file__).parent.parent.absolute()
#         / "data"
#         / "effective_area_IDL_testing_files"
#     )
#     filter_data_files = directory.glob("**/*.txt")
#     return sorted(filter_data_files)
import numpy as np
def get_IDL_data_files():
    data_dir = Path(__file__).parent / "data" / "effective_area_IDL_testing_files"
    assert data_dir.exists(), f"Data directory {data_dir} does not exist."
    return sorted(data_dir.glob("**/*.txt"))  # Explicitly return the list of files

# def get_IDL_data_files():
#     data_dir = Path(__file__).parent / "data" / "effective_area_IDL_testing_files" 
#     assert data_dir.exists(), f"Data directory {data_dir} does not exist."
#     files = sorted(data_dir.glob("**/*.txt")) 


filenames = get_IDL_data_files()


def _IDL_raw_data_list(filename):
    with open(filename) as filter_file:  # noqa: PTH123
        list_of_IDL_effective_area_data = []
        for line in filter_file:
            stripped_line = line.strip()
            line_list = stripped_line.split()
            list_of_IDL_effective_area_data.append(line_list)

    return list_of_IDL_effective_area_data


def IDL_test_filter_name(list_of_lists):
    return str(list_of_lists[0][1])


def IDL_test_date(list_of_lists):
    obs_date = str(list_of_lists[1][1])
    obs_time = str(list_of_lists[1][2])

    day = int(obs_date[:2])

    month_datetime_object = datetime.strptime(obs_date[3:6], "%b")
    month = month_datetime_object.month

    year = int(obs_date[8:12])

    hour = int(obs_time[:2])
    minute = int(obs_time[3:5])
    second = int(obs_time[6:8])

    return datetime(year, month, day, hour, minute, second)


def _IDL_effective_area_raw_data(filename):
    with open(filename) as filter_file:  # noqa: PTH123
        list_of_lists = []
        for line in filter_file:
            stripped_line = line.strip()
            line_list = stripped_line.split()
            list_of_lists.append(line_list)

    effective_area = [list_of_lists[i][1] for i in range(3, len(list_of_lists))]
    effective_area = [float(i) for i in effective_area] * u.cm**2

    return effective_area


# @pytest.mark.parametrize("filename", filenames)
# def test_EffectiveAreaPreparatory_effective_area(filename, allclose):
#     data_list = _IDL_raw_data_list(filename)

#     filter_name = IDL_test_filter_name(data_list)
#     filter_obs_date = IDL_test_date(data_list)

#     IDL_effective_area = _IDL_effective_area_raw_data(filename)

#     instance = EffectiveAreaFundamental(filter_name, filter_obs_date)
#     actual_effective_area = instance.effective_area()

#     assert actual_effective_area.unit == IDL_effective_area.unit
#     assert allclose(actual_effective_area.value, IDL_effective_area.value, atol=1e-2)
import matplotlib.pyplot as plt
from pathlib import Path
import xrtpy

OUTPUT_DIR = Path("effective_area_plots")
OUTPUT_DIR.mkdir(exist_ok=True)  # Create the main output directory if it doesn't exist

@pytest.mark.parametrize("filename", get_IDL_data_files())
def test_effective_area_compare_idl(filename):
    print(f"\n\nTesting file: {filename}\n")

    # Read the filter name and observation date from the file
    with filename.open() as f:
        filter_name = f.readline().split()[1]
        filter_obs_date = " ".join(f.readline().split()[1:])
        print(f"Filter name: {filter_name}, Observation date: {filter_obs_date}")  # Debugging output

    # Correct non-standard date format
    filter_obs_date = filter_obs_date.replace("Sept", "Sep")

    # Load IDL data from the file
    IDL_data = np.loadtxt(filename, skiprows=3)
    IDL_wavelength = IDL_data[:, 0] * u.AA
    IDL_effective_area = IDL_data[:, 1] * u.cm**2

    # Compute effective area using XRTpy
    instance = xrtpy.response.EffectiveAreaFundamental(filter_name, filter_obs_date)
    actual_effective_area = instance.effective_area()

    # Interpolate XRTpy effective area onto the IDL wavelength grid
    XRTpy_effective_area_interp = np.interp(
        IDL_wavelength.value,  # Target grid (IDL wavelengths)
        instance.channel_wavelength.value,  # Source grid (XRTpy wavelengths)
        actual_effective_area.value  # Data to interpolate (XRTpy effective area)
    )

    # Make both arrays dimensionless for comparison
    IDL_effective_area_unitless = IDL_effective_area.value
    XRTpy_effective_area_unitless = XRTpy_effective_area_interp

    # Compare effective areas using relative tolerance (unitless comparison)
    rtol = 1e-4
    differences = np.abs(XRTpy_effective_area_unitless - IDL_effective_area_unitless)
    max_diff = np.max(differences)
    failed_indices = np.where(differences > rtol * np.abs(IDL_effective_area_unitless))[0]

    if failed_indices.size > 0:
        print(f"Test failed for filter {filter_name} on {filter_obs_date}")
        print(f"Max difference: {max_diff}")

        # Save the plot
        save_effective_area_plot(
            IDL_wavelength.value,
            IDL_effective_area_unitless,
            XRTpy_effective_area_unitless,
            failed_indices,
            filter_name,
            filter_obs_date,
        )

        # Fail the test
        assert False, f"Effective areas differ for filter {filter_name} on {filter_obs_date}"

def save_effective_area_plot(wavelength, IDL_area, XRTpy_area, failed_indices, filter_name, obs_date):
    """
    Saves the effective area comparison plot to a subfolder for the given filter.
    """
    # Create a subdirectory for the filter
    filter_dir = OUTPUT_DIR / filter_name
    filter_dir.mkdir(parents=True, exist_ok=True)

    # Define the output filename
    output_file = filter_dir / f"{filter_name}_{obs_date.replace(':', '').replace(' ', '_')}.png"

    # Count the number of failed points
    num_failed_points = len(failed_indices)

    # Plot the data
    plt.figure(figsize=(10, 6))
    plt.plot(wavelength, IDL_area, label="IDL Effective Area", color="blue", lw=2)
    plt.plot(wavelength, XRTpy_area, label="XRTpy Effective Area", color="green", linestyle="--", lw=2)
    plt.scatter(
        wavelength[failed_indices],
        IDL_area[failed_indices],
        color="red",
        label=f"Failed Points: {num_failed_points}",
        zorder=5,
    )
    plt.xlabel("Wavelength (Å)", fontsize=14)
    plt.ylabel("Effective Area (cm²)", fontsize=14)
    plt.title(
        f"Effective Area Comparison for {filter_name} on {obs_date} (Quadratic)",
        fontsize=16,
    )
    plt.legend(fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()

    # Save the plot
    plt.savefig(output_file, dpi=300)
    plt.close()
    print(f"Saved plot to {output_file}")

def save_combined_filter_plots(filter_name, test_data):
    """
    Saves a combined plot with subplots for all dates associated with a single filter.
    
    Parameters:
        filter_name (str): The name of the filter.
        test_data (list of tuples): Each tuple contains (wavelength, IDL_area, XRTpy_area, failed_indices, obs_date).
    """
    # Create a subdirectory for the filter
    filter_dir = OUTPUT_DIR / filter_name
    filter_dir.mkdir(parents=True, exist_ok=True)

    # Define the output filename
    output_file = filter_dir / f"{filter_name}_combined.png"

    # Determine subplot grid size
    num_tests = len(test_data)
    num_cols = 3
    num_rows = (num_tests + num_cols - 1) // num_cols  # Ceiling division for rows

    # Create the figure and axes
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 5 * num_rows), squeeze=False)

    for i, (wavelength, IDL_area, XRTpy_area, failed_indices, obs_date) in enumerate(test_data):
        row, col = divmod(i, num_cols)
        ax = axes[row, col]

        # Plot data
        ax.plot(wavelength, IDL_area, label="IDL Effective Area", color="blue", lw=2)
        ax.plot(wavelength, XRTpy_area, label="XRTpy Effective Area", color="green", linestyle="--", lw=2)
        ax.scatter(
            wavelength[failed_indices],
            IDL_area[failed_indices],
            color="red",
            label=f"Failed Points: {len(failed_indices)}",
            zorder=5,
        )

        # Customize subplot
        ax.set_title(f"{filter_name} - {obs_date} (Quadratic)", fontsize=12)
        ax.set_xlabel("Wavelength (Å)", fontsize=10)
        ax.set_ylabel("Effective Area (cm²)", fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.7)
        ax.legend(fontsize=8)

    # Remove empty subplots
    for j in range(i + 1, num_rows * num_cols):
        fig.delaxes(axes[j // num_cols, j % num_cols])

    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()
    print(f"Saved combined plot to {output_file}")


# Example Usage in Your Test
filter_test_data = {}  # Dictionary to store data for each filter

@pytest.mark.parametrize("filename", get_IDL_data_files())
def test_effective_area_compare_idl(filename):
    print(f"\n\nTesting file: {filename}\n")

    # Read the filter name and observation date from the file
    with filename.open() as f:
        filter_name = f.readline().split()[1]
        filter_obs_date = " ".join(f.readline().split()[1:])
        print(f"Filter name: {filter_name}, Observation date: {filter_obs_date}")  # Debugging output

    # Correct non-standard date format
    filter_obs_date = filter_obs_date.replace("Sept", "Sep")

    # Load IDL data from the file
    IDL_data = np.loadtxt(filename, skiprows=3)
    IDL_wavelength = IDL_data[:, 0] * u.AA
    IDL_effective_area = IDL_data[:, 1] * u.cm**2

    # Compute effective area using XRTpy
    instance = xrtpy.response.EffectiveAreaFundamental(filter_name, filter_obs_date)
    actual_effective_area = instance.effective_area()

    # Interpolate XRTpy effective area onto the IDL wavelength grid
    XRTpy_effective_area_interp = np.interp(
        IDL_wavelength.value,  # Target grid (IDL wavelengths)
        instance.channel_wavelength.value,  # Source grid (XRTpy wavelengths)
        actual_effective_area.value  # Data to interpolate (XRTpy effective area)
    )

    # Make both arrays dimensionless for comparison
    IDL_effective_area_unitless = IDL_effective_area.value
    XRTpy_effective_area_unitless = XRTpy_effective_area_interp

    # Compare effective areas using relative tolerance (unitless comparison)
    rtol = 1e-4
    differences = np.abs(XRTpy_effective_area_unitless - IDL_effective_area_unitless)
    max_diff = np.max(differences)
    failed_indices = np.where(differences > rtol * np.abs(IDL_effective_area_unitless))[0]

    # Save individual plots
    save_effective_area_plot(
        IDL_wavelength.value,
        IDL_effective_area_unitless,
        XRTpy_effective_area_unitless,
        failed_indices,
        filter_name,
        filter_obs_date,
    )

    # Aggregate data for combined plotting
    if filter_name not in filter_test_data:
        filter_test_data[filter_name] = []
    filter_test_data[filter_name].append(
        (IDL_wavelength.value, IDL_effective_area_unitless, XRTpy_effective_area_unitless, failed_indices, filter_obs_date)
    )

    # Fail the test if there are failed indices
    assert failed_indices.size == 0, f"Effective areas differ for filter {filter_name} on {filter_obs_date}"


# After all tests, generate combined plots for each filter
@pytest.fixture(scope="session", autouse=True)
def create_combined_plots():
    yield  # Ensure tests complete first
    for filter_name, test_data in filter_test_data.items():
        save_combined_filter_plots(filter_name, test_data)