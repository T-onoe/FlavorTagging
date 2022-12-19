# FlavorTagging (under construction)
This is an algorithm for Flavor Tagging on ILC(International Linear Collider).

Since the system is not open to the public now, it is intended to be implemented on the data server of Kyushu University.

If you are a member of Kyushu University's Experimental Particle Physics Laboratory, please login to the bepp server and run it.

------------------------------------

@How to run
The first time you run code/train_flav_trk.py, it will start creating datasets.

"python train_flav_trk.py"

When the number of data you want to create is reached, 
change idx_max in processed_file_names in dataset_flv_trk.py to the desired number and execute.

"idx_max = (number you want)"

Then, run again.

--------------------------------------

Output is saved in "output" directory.

--------------------------------------

If you have any questions, please contact me at the e-mail address below.

onoe@epp.phys.kyushu-u.ac.jp
