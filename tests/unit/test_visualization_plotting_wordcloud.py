"""Tests for word cloud generation functionality."""

import matplotlib.pyplot as plt
import numpy as np
import pytest
from pathlib import Path
from PIL import Image

from PMRTN.visualization.plotting import generate_wordcloud, PlottingError


@pytest.fixture
def sample_text():
    """Create sample Spanish text."""
    return """
    España economía mercado valores acciones bolsa empresas inversión
    financiero banco sectores crecimiento análisis resultados beneficios
    índice cotización tendencia operaciones negocio estrategia desarrollo
    país industria comercio exportación importación producción tecnología
    inversores capital riesgo rentabilidad dividendos acciones mercados
    """


@pytest.fixture
def sample_text_list():
    """Create list of sample texts."""
    return [
        "España economía mercado valores",
        "acciones bolsa empresas inversión",
        "financiero banco sectores crecimiento"
    ]


@pytest.fixture
def simple_mask(tmp_path):
    """Create simple square mask image."""
    mask_path = tmp_path / "mask.png"
    # Create a simple white square with black border
    img_array = np.ones((100, 100, 3), dtype=np.uint8) * 255
    img_array[10:90, 10:90] = 0  # Black square in center
    img = Image.fromarray(img_array)
    img.save(mask_path)
    return mask_path


class TestGenerateWordcloud:
    """Tests for generate_wordcloud function."""
    
    def test_basic_wordcloud(self, sample_text):
        """Test basic word cloud generation."""
        fig = generate_wordcloud(
            sample_text,
            show_plot=False
        )
        
        assert fig is not None
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
    
    def test_with_title(self, sample_text):
        """Test word cloud with custom title."""
        fig = generate_wordcloud(
            sample_text,
            title="Test Word Cloud",
            show_plot=False
        )
        
        assert fig is not None
        plt.close(fig)
    
    def test_from_text_list(self, sample_text_list):
        """Test word cloud from list of texts."""
        fig = generate_wordcloud(
            sample_text_list,
            show_plot=False
        )
        
        assert fig is not None
        plt.close(fig)
    
    def test_with_colormap(self, sample_text):
        """Test different colormaps."""
        for colormap in ['viridis', 'plasma', 'Blues', 'Reds']:
            fig = generate_wordcloud(
                sample_text,
                colormap=colormap,
                show_plot=False
            )
            assert fig is not None
            plt.close(fig)
    
    def test_max_words_parameter(self, sample_text):
        """Test max_words parameter."""
        for max_words in [50, 100, 200]:
            fig = generate_wordcloud(
                sample_text,
                max_words=max_words,
                show_plot=False
            )
            assert fig is not None
            plt.close(fig)
    
    def test_custom_dimensions(self, sample_text):
        """Test custom width and height."""
        fig = generate_wordcloud(
            sample_text,
            width=1200,
            height=600,
            show_plot=False
        )
        
        assert fig is not None
        plt.close(fig)
    
    def test_custom_background_color(self, sample_text):
        """Test custom background colors."""
        for bg_color in ['white', 'black', 'lightgray']:
            fig = generate_wordcloud(
                sample_text,
                background_color=bg_color,
                show_plot=False
            )
            assert fig is not None
            plt.close(fig)
    
    def test_custom_stopwords(self, sample_text):
        """Test with custom stopwords."""
        custom_stopwords = {'españa', 'economía', 'mercado'}
        
        fig = generate_wordcloud(
            sample_text,
            stopwords=custom_stopwords,
            show_plot=False
        )
        
        assert fig is not None
        plt.close(fig)
    
    def test_save_output(self, sample_text, tmp_path):
        """Test saving word cloud to file."""
        output_path = tmp_path / "wordcloud.pdf"
        
        fig = generate_wordcloud(
            sample_text,
            output_path=output_path,
            save_output=True,
            show_plot=False
        )
        
        assert output_path.exists()
        assert output_path.stat().st_size > 0
        plt.close(fig)
    
    def test_save_to_directory(self, sample_text, tmp_path):
        """Test saving with directory path (should use title for filename)."""
        fig = generate_wordcloud(
            sample_text,
            title="Test Cloud",
            output_path=tmp_path,
            save_output=True,
            show_plot=False
        )
        
        expected_path = tmp_path / "Test_Cloud.pdf"
        assert expected_path.exists()
        plt.close(fig)
    
    def test_with_mask_image_file(self, sample_text, simple_mask):
        """Test word cloud with mask from image file."""
        fig = generate_wordcloud(
            sample_text,
            mask=str(simple_mask),
            show_plot=False
        )
        
        assert fig is not None
        plt.close(fig)
    
    def test_with_mask_numpy_array(self, sample_text):
        """Test word cloud with mask as numpy array."""
        # Create a simple circular mask
        x, y = np.ogrid[:300, :300]
        mask_array = ((x - 150) ** 2 + (y - 150) ** 2 > 130 ** 2).astype(np.uint8) * 255
        
        fig = generate_wordcloud(
            sample_text,
            mask=mask_array,
            show_plot=False
        )
        
        assert fig is not None
        plt.close(fig)
    
    def test_custom_figsize(self, sample_text):
        """Test custom figure size."""
        fig = generate_wordcloud(
            sample_text,
            figsize=(15, 10),
            show_plot=False
        )
        
        assert fig.get_figwidth() == 15
        assert fig.get_figheight() == 10
        plt.close(fig)
    
    def test_custom_font_sizes(self, sample_text):
        """Test custom font parameters."""
        fig = generate_wordcloud(
            sample_text,
            title_fontsize=20,
            title_pad=25,
            min_font_size=8,
            show_plot=False
        )
        
        assert fig is not None
        plt.close(fig)
    
    def test_relative_scaling(self, sample_text):
        """Test relative scaling parameter."""
        for scaling in [0.0, 0.5, 1.0]:
            fig = generate_wordcloud(
                sample_text,
                relative_scaling=scaling,
                show_plot=False
            )
            assert fig is not None
            plt.close(fig)
    
    def test_empty_string_raises_error(self):
        """Test that empty string raises PlottingError."""
        with pytest.raises(PlottingError, match="Text is empty"):
            generate_wordcloud("", show_plot=False)
    
    def test_whitespace_only_raises_error(self):
        """Test that whitespace-only string raises PlottingError."""
        with pytest.raises(PlottingError, match="Text is empty"):
            generate_wordcloud("   \n  \t  ", show_plot=False)
    
    def test_invalid_type_raises_error(self):
        """Test that invalid input type raises PlottingError."""
        with pytest.raises(PlottingError, match="must be a string or list of strings"):
            generate_wordcloud(12345, show_plot=False)
    
    def test_invalid_mask_type_raises_error(self, sample_text):
        """Test that invalid mask type raises PlottingError."""
        with pytest.raises(PlottingError, match="Mask must be"):
            generate_wordcloud(
                sample_text,
                mask=12345,
                show_plot=False
            )
    
    def test_nonexistent_mask_file_raises_error(self, sample_text):
        """Test that nonexistent mask file raises PlottingError."""
        with pytest.raises(PlottingError, match="Failed to load mask image"):
            generate_wordcloud(
                sample_text,
                mask="nonexistent_file.png",
                show_plot=False
            )
    
    def test_spanish_stopwords_applied(self, sample_text):
        """Test that Spanish stopwords are filtered out."""
        # Generate word cloud (should filter Spanish stopwords by default)
        fig = generate_wordcloud(
            sample_text,
            max_words=50,
            show_plot=False
        )
        
        assert fig is not None
        plt.close(fig)
    
    def test_realistic_financial_text(self):
        """Test with realistic Spanish financial news text."""
        text = """
        La bolsa española registró hoy una jornada positiva impulsada por
        el sector bancario y las empresas tecnológicas. El IBEX 35 cerró
        con ganancias del 1.2% gracias al buen comportamiento de los valores
        financieros. Los analistas destacan el crecimiento económico y la
        mejora de los resultados empresariales como factores clave para
        explicar esta tendencia alcista en los mercados de valores.
        """
        
        fig = generate_wordcloud(
            text,
            title="Noticias Financieras",
            colormap='Blues',
            max_words=100,
            show_plot=False
        )
        
        assert fig is not None
        plt.close(fig)
    
    def test_long_text_performance(self):
        """Test performance with long text."""
        # Create long text by repeating
        text = """
        España economía mercado valores acciones bolsa empresas inversión
        financiero banco sectores crecimiento análisis resultados beneficios
        """ * 100
        
        fig = generate_wordcloud(
            text,
            max_words=200,
            show_plot=False
        )
        
        assert fig is not None
        plt.close(fig)
    
    def test_with_mask_and_colormap(self, sample_text, simple_mask):
        """Test word cloud with both mask and colormap."""
        fig = generate_wordcloud(
            sample_text,
            mask=str(simple_mask),
            colormap='viridis',
            show_plot=False
        )
        
        assert fig is not None
        plt.close(fig)
    
    def test_empty_list_raises_error(self):
        """Test that empty list raises error after joining."""
        with pytest.raises(PlottingError, match="Text is empty"):
            generate_wordcloud([], show_plot=False)
    
    def test_list_of_empty_strings_raises_error(self):
        """Test that list of empty strings raises error."""
        with pytest.raises(PlottingError, match="Text is empty"):
            generate_wordcloud(["", "  ", "\n"], show_plot=False)
