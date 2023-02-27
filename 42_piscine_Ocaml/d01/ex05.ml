let rec
	hfs_m (m: int): int =
		match m with
		| m when m > 0 -> m - hfs_f ( hfs_m (m - 1) )
		| m when m = 0 -> 0
		| _            -> -1
and
	hfs_f (f: int): int =
		match f with
		| f when f > 0 -> f - hfs_m ( hfs_f (f - 1) )
		| f when f = 0 -> 1
		| _            -> -1
;;

assert ( hfs_m 0 = 0 ) ;;
assert ( hfs_f 0 = 1 ) ;;
assert ( hfs_m 4 = 2 ) ;;
assert ( hfs_f 4 = 3 ) ;;
