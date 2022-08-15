from argparse import ArgumentParser

from pyradise.conversion import (
    DicomSubjectDirectoryCrawler,
    DicomSubjectConverter,
    RTSSToImageConverter,
    DicomSeriesRTStructureSetInfo,
    DicomSeriesImageInfo,
    DicomSeriesRegistrationInfo,
    load_datasets
)


def main(input_directory_path: str) -> None:
    crawler = DicomSubjectDirectoryCrawler(input_directory_path)
    series_info = crawler.execute()

    result_1 = DicomSubjectConverter(series_info).convert()

    rtss_info = [entry for entry in series_info if isinstance(entry, DicomSeriesRTStructureSetInfo)][0]
    image_infos = [entry.path for entry in series_info if isinstance(entry, DicomSeriesImageInfo)]
    # image_datasets = load_datasets(image_infos[0])
    image_datasets = []
    [image_datasets.extend(load_datasets(info)) for info in image_infos]

    # registration_info = [entry.dataset for entry in series_info if isinstance(entry, DicomSeriesRegistrationInfo)][0]
    result_2 = RTSSToImageConverter(rtss_info.dataset, tuple(image_datasets)).convert()

    print('Test it!')


if __name__ == '__main__':
    main('D:/DataBackups/2020_08_ISAS_OAR_work/ISAS_GBM_001')
