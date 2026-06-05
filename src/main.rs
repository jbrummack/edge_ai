use edge_ai::embedders::vision::vmodel::VModel;

fn main() {
    unsafe {
        executorch::platform::pal_init();
    }

    let mut model = VModel::<448>::new("./models/tips_v2_coreml_q8.pte").unwrap();
    let image = image::open("./models/red-apple.jpg").unwrap();
    let emb = model.embed(image);
    println!("{emb:?}");
}
