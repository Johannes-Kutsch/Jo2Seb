from tqdm.contrib.concurrent import thread_map
from itertools import product
import python.Utils.Download
import python.Utils.FilePaths

def download_gfw_voyages_parallel(token, years=range(2017, 2026), months=range(1, 13), prefix="voyages", max_workers=6):
    """
    Downloads GFW voyages data in parallel

    :param token: Needs to be a valid GFW token and is not the api token. You can find it by downloading a voyage file from the Website and looking at the Authorization header. Token is valid for ~15 minutes.
    :param years: Year range to download
    :param months: Month range to download
    :param prefix: filename prefix
    :param max_workers: Maximum number of parallel downloads
    :return:
    """
    base_url = "https://gateway.api.globalfishingwatch.org/v3/download/datasets/public-voyages-confidence-4:v20220922/download"
    headers = {"Authorization": f"Bearer {token}"}

    all_files = [(year, month) for year, month in product(years, months)]

    def download_wrapper(year_month):
        year, month = year_month
        file_name = f"{prefix}_{year}{month:02d}.csv"
        signed_url = python.Utils.Download.get_signed_url(base_url, headers, file_name)
        if signed_url is None:
            return f"No signe URL for {file_name}"
        python.Utils.Download.download_file_from_signed_url(signed_url, file_name, python.Utils.FilePaths.VOYAGES_DIR)
        return f"Finished download of {file_name}"

    results = thread_map(download_wrapper, all_files, max_workers=max_workers, desc="Downloading files")
    for r in results:
        print(r)