from argparse import ArgumentParser

from pyradise.conversion.crawling import DicomSubjectDirectoryCrawler


def main(subject_directory: str) -> None:
    print(f'Crawling at {subject_directory}...')
    crawler = DicomSubjectDirectoryCrawler(subject_directory, write_modality_config=True)
    crawler.execute()
    print('Crawling finished!')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-subject_directory', type=str)
    args = parser.parse_args()

    main(args.subject_directory)
