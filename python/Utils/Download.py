import requests

def get_signed_url(base_url, headers, file_name):
    """
    Get a signed URL to download a file from a base url.

    :param base_url: Base URL to get the signed URL from.
    :param headers: Headers to include in the request.
    :param file_name: Name of the file to download.
    :return: Signed URL if successful, None otherwise.
    """
    resp = requests.get(base_url, headers=headers, params={"file-path": file_name})
    if resp.status_code != 200:
        print(f"Error for {file_name}: {resp.status_code}")
        return None

    signed_url = resp.json().get("url")
    if not signed_url:
        print(f"No url in response for {file_name}")
        return None

    return signed_url


def download_file_from_signed_url(signed_url, file_name, save_directory):
    """
    Downloads a file from a signed URL using requests.

    :param signed_url: The signed URL to download the file from.
    :param file_name: The name of the file to save.
    :param save_directory: The directory to save the file in.
    """
    file_location = save_directory / file_name
    try:
        with requests.get(signed_url, stream=True) as r:
            if r.status_code == 404:
                print(f"File not Found (404): {file_name}")
                return
            r.raise_for_status()
            with open(file_location, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)

        print(f"File saved: {file_location}")

    except requests.HTTPError as e:
        print(f"HTTP-Error when downloading {file_name}: {e}")
    except requests.RequestException as e:
        print(f"Network/Request error when downloading {file_name}: {e}")
    except Exception as e:
        print(f"Unexpected Error when downloading {file_name}: {e}")