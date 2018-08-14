from setuptools import setup

setup(name="fcs1",
      version="0.0.1",
      description="FCS automatic analysis.",
      url="https://github.com/bankanidhi/fcs1",
      entry_points={
          "console_scripts": [
              "fcs1=fcs1.run:run"
          ]
      },
      install_requires=["numpy==1.14.5",
                        "scipy==1.1.0",
                        "matplotlib==2.2.2",
                        "lmfit==0.9.11"],
      license="MIT",
      zip_safe=False)
