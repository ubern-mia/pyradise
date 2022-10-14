from argparse import ArgumentParser
from typing import Tuple

from pyradise.conversion import DicomSubjectDirectoryCrawler, DicomSeriesInfo, DicomSubjectConverter
from pyradise.serialization import SubjectWriter


def main(subject_directory: str) -> None:
    crawler = DicomSubjectDirectoryCrawler(subject_directory)
    series_infos: Tuple[DicomSeriesInfo] = crawler.execute()

    for series_info in series_infos:
        print(f'Retrieved series {series_info.series_description} '
              f'with SeriesInstanceUID {series_info.series_instance_uid}')

    converter = DicomSubjectConverter(series_infos)
    subject = converter.convert()

    output_dir = 'D:/temp/output_test'
    SubjectWriter().write(output_dir, subject, False)

    print(f'Subject name: {subject.get_name()}')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-subject_directory', type=str, default='D:/temp/dicom_test_data/ISAS_GBM_001')
    args = parser.parse_args()

    main(args.subject_directory)
