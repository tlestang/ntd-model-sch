from amis_integration import adapt_parameters_to_shape
import pandas as pd
import pandas.testing as pdt

def test_adapt_parameters_to_shape():
    example_parameter_sets = (
        [0.5, 0.25],
        [1, 2]
    )
    data_frame = adapt_parameters_to_shape(example_parameter_sets)
    pdt.assert_frame_equal(data_frame, pd.DataFrame([[0.5, 1], [0.25, 2]],
                  columns=['R0', 'k']))