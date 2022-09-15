import setuptools

setuptools.setup(
    name='sch_simulation',
    version='0.0.1',
    url='https://www.ntdmodelling.org',
    maintainer='ArtRabbit',
    maintainer_email='support@artrabbit.com',
    description='SCH simulation model',
    long_description='Individual-based model in Medley 1989 thesis and Anderson&Medley 1985.',
    packages=setuptools.find_packages(),
    python_requires='>=3.6',
    install_requires=['numpy', 'scipy', 'pandas==1.4.3', 'joblib', 'matplotlib', 'openpyxl'],
    include_package_data=True
)
