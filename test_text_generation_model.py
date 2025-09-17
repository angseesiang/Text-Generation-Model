import unittest
from text_generation_model import TextGenerationModel

class TestTextGenerationModel(unittest.TestCase):
    def setUp(self):
        # Initialize the model for testing
        self.model = TextGenerationModel()

    def test_generate_text(self):
        # Test text generation
        prompt = "Once upon a time"
        generated_text = self.model.generate_text(prompt)
        self.assertIsInstance(generated_text, str)
        self.assertTrue(len(generated_text) > len(prompt))

    def test_save_load_model(self):
        # Test saving and Loading the model
        self.model.save_model('test_model')
        loaded_model = TextGenerationModel()
        loaded_model.load_model('test_model')
        prompt = "Once upon a time"
        generated_text = loaded_model.generate_text(prompt)
        self.assertIsInstance(generated_text, str)
        self.assertTrue(len(generated_text) > len(prompt))


if __name__ == '__main__':
    unittest.main()

