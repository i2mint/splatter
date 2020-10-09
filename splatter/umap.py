"""Umap Splatters
To work you'll need:

pip install numba==0.50.1  # 0.51.2 might work, but 0.50.1 safer
pip install umap-learn
pip install matplotlib
pip install datashader
pip install bokeh
pip install pandas
pip install holoviews
pip install colorcet
"""

# from oui.util import ModuleNotFoundIgnore
# with ModuleNotFoundIgnore():
#     ...

try:
    import umap
    import umap.plot
    from numba.core.typing import cffi_utils

except Exception as err:
    print(f"If you get a TypeError: wrapped() missing 1 required positional argument: 'name', "
          f"you might want to consider doing a ``pip install numba==0.50.1``")
    raise


# from numba.core.typing import cffi_utils
# import pandas as pd

def plot_umap_of_xy(X, y=None, return_projection=False, **umap_plot_points_kwargs):
    model = umap.UMAP()
    model.fit(X)
    t = umap.plot.points(model, labels=y, **umap_plot_points_kwargs)
    if not return_projection:
        return t
    else:
        projection = model.transform(X)
        return t, projection
