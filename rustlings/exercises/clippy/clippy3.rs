// clippy3.rs
// Here's a couple more easy Clippy fixes, so you can see its utility.



#[allow(unused_variables, unused_assignments)]
fn main()
{
	let my_option: Option<()> = None;
	if my_option.is_some()
	{
		println!("my_option is some");
	}

	let my_arr = &[
		-1, -2, -3,
		-4, -5, -6,
	];
	println!("My array! Here it is: {:?}", my_arr);

	let mut vec = vec![1, 2, 3, 4, 5];
	vec.clear();
	println!("This Vec is not empty, see? {:?}", vec);

	let mut value_a = 45;
	let mut value_b = 66;
	// Let's swap these two!
	(value_a, value_b) = (value_b, value_a);
	println!("value a: {}; value b: {}", value_a, value_b);
}
