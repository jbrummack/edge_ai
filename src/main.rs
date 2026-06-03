use edge_ai::embedders::vision::vmodel::VModel;

fn main() {
    let mut model = VModel::<448>::new("").unwrap();
    let image = image::open("./models/red-apple.jpg").unwrap();
    let emb = model.embed(image);
    println!("{emb:?}");
}
