import time
import pandas as pd

def denoument(outfile, polyinfo_df, t_start):
	# Create and print dataframe to CSV
	polyinfo_df.to_csv(outfile + '.csv')

	total_points = len(polyinfo_df)
	total_time = time.time() - t_start
	print 'Success!'
	print 'Excecution time: %.2f min for %i data points' % \
			(total_time/60, total_points)