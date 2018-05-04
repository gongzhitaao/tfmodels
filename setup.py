from setuptools import setup


def readme():
    with open('README.md') as f:
        return f.read()


setup(name='tfmodels',
      version='0.1',
      description='I do not understand things I can not build.',
      long_description=readme(),
      classifiers=[
          'Development Status :: 3 - Alpha',
          'Intended Audience :: Science/Research'
          'License :: OSI Approved :: MIT License',
          'Natural Language :: English',
          'Operating System :: POSIX :: Linux',
          'Programming Language :: Python :: 3 :: Only',
          'Topic :: Scientific/Engineering :: Artificial Intelligence'],
      keywords='tensorflow, artificial intelligence,',
      url='https://github.com/gongzhitaao/tfmodels',
      author='Zhitao Gong',
      author_email='zhitaao.gong@gmail.com',
      license='MIT',
      packages=['tfmodels'],
      install_requires=[
          'bleach',
          'chardet',
          'gensim',
          'nltk',
          'numpy',
          'tensorflow',
          'tqdm'
      ],
      test_suite='nose.collector',
      tests_require=['nose', 'nose-cover3'],
      include_package_data=True,
      zip_safe=False)
