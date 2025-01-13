import pygame
import sys

# Initialize Pygame
pygame.init()

# Constants
GRID_SIZE = 4  # Grid size (4x4)
CELL_SIZE = 100  # Cell size in pixels
MARGIN = 2  # Margin between cells
SCREEN_SIZE = GRID_SIZE * CELL_SIZE + (GRID_SIZE - 1) * MARGIN
FONT_SIZE = 36

# Create screen
screen = pygame.display.set_mode((SCREEN_SIZE, SCREEN_SIZE))
pygame.display.set_caption("Single Tile Drawer")

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (200, 200, 200)

# Font for tile values
font = pygame.font.Font(None, FONT_SIZE)

# Draw single tile
def draw_tile(value, row, col):
    """Draw a single tile at the specified row and column with the given value."""
    x = col * (CELL_SIZE + MARGIN)
    y = row * (CELL_SIZE + MARGIN)
    
    # Draw the tile rectangle
    pygame.draw.rect(screen, GRAY, (x, y, CELL_SIZE, CELL_SIZE))
    
    # Draw the value, if not None
    if value is not None:
        text = font.render(str(value), True, BLACK)
        text_rect = text.get_rect(center=(x + CELL_SIZE / 2, y + CELL_SIZE / 2))
        screen.blit(text, text_rect)

# Main game loop
def main():
    clock = pygame.time.Clock()

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        # Fill the screen
        screen.fill(WHITE)
        
        # Draw a single tile for demonstration
        draw_tile(8, 0.5,0.5)  # Draw tile with value 8 at row 1, column 2
        
        # Update the display
        pygame.display.flip()
        clock.tick(30)

# Run the program
main()
