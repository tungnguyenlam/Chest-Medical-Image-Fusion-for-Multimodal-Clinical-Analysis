from idc_index import index

client = index.IDCClient()

# Download the entire Lung-PET-CT-Dx collection
client.download_from_selection(
    collection_id="lung_pet_ct_dx", downloadDir="./data/Lung-PET-CT-Dx/"
)


# from tcia_utils import nbia

# from nbiatoolkit import NBIAClient

# from tqdm.auto import tqdm


# def main():
#     with NBIAClient() as client:
#         # First, get series UIDs from your manifest
#         import re

#         with open(
#             "./data/data-tcia-download/Lung-PET-CT-Dx-NBIA-Manifest-122220.tcia"
#         ) as f:
#             content = f.read()
#         # Extract UIDs (they appear after the manifest header lines)

#         uids = [
#             line.strip()
#             for line in content.splitlines()
#             if re.match(r"^\d+\.\d+", line.strip())
#         ]

#         print(f"Found {len(uids)} series to download")

#         client.downloadSeries(
#             SeriesInstanceUID=uids,
#             downloadDir="./data/Lung-PET-CT-Dx/",
#             nParallel=10,  # parallel downloads
#             Progressbar=True,
#         )


# if __name__ == "__main__":
#     main()

# # Download all series from your manifest file
# nbia.downloadSeries(
#     series_data="./data/data-tcia-download/Lung-PET-CT-Dx-NBIA-Manifest-122220.tcia",
#     input_type="manifest",
#     path="./data/Lung-PET-CT-Dx/",
#     threads=8,
# )
