from idc_index import index

client = index.IDCClient()

client.download_from_selection(
    collection_id="nsclc_radiogenomics",
    downloadDir="./data/NSCLC-Radiogenomics/"
)