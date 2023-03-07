let read_file (filename: string): string array =
	let lines = ref [||] in
	let in_channel = open_in (filename) in
	try
	(
		while true;
		do
			let new_value = [| input_line (in_channel)|] in
			lines := Array.append (!lines) (new_value)
		done;
		!lines
	)
	with End_of_file ->
	(
		close_in in_channel;
		!lines
	)
;;



let args = Sys.argv ;;
if Array.length (args) != 2 then raise (Failure ("expects single filename arg")) else ();;
let filename = args.(1) ;;
let jokes = read_file (filename);;

Random.self_init() ;;

let joke = jokes.(Random.int (Array.length jokes)) ;;
print_endline(joke);;
