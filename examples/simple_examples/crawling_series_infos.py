from argparse import ArgumentParser
from typing import Tuple

from pyradise.conversion.crawling import DicomSubjectDirectoryCrawler
from pyradise.conversion.series_information import DicomSeriesInfo


def main(subject_directory: str) -> None:
    crawler = DicomSubjectDirectoryCrawler(subject_directory)
    series_infos: Tuple[DicomSeriesInfo] = crawler.execute()

    for series_info in series_infos:
        print(f'Retrieved series {series_info.series_description} '
              f'with SeriesInstanceUID {series_info.series_instance_uid}')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-subject_directory', type=str)
    args = parser.parse_args()

    main(args.subject_directory)
