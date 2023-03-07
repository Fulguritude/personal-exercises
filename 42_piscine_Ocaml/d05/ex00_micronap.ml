let my_sleep () = Unix.sleep 1;;

let args = Sys.argv ;;
if Array.length (args) != 2 then raise (Failure ("expects single int arg")) else ();;
let count = int_of_string(args.(1)) ;;

for i = 1 to count do
	print_endline("Sleeping... #" ^ string_of_int(i));
	my_sleep ();
done;;
