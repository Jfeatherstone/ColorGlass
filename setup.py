import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
     name='colorglass',  
     version='1.0',
     author="Jack Featherstone, Vladimir Skokov",
     author_email="jdfeathe@ncsu.edu",
     description="Simulation code for color glass condensates in dense-dilute approximation.",
     long_description=long_description,
     long_description_content_type="text/markdown",
     url="https://github.com/Jfeatherstone/ColorGlass",
     packages=setuptools.find_packages(),
     classifiers=[
         "Programming Language :: Python :: 3",
         "License :: OSI Approved :: MIT License",
         "Operating System :: OS Independent",
     ],
 )
