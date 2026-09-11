
#include <SFML/Graphics.hpp>
#include <SFML/Window.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <optional>
#include <vector>

// ------------------------------------------------------------
// Configuration
// ------------------------------------------------------------

constexpr unsigned int WINDOW_WIDTH  = 900;
constexpr unsigned int WINDOW_HEIGHT = 700;

constexpr unsigned int CANVAS_WIDTH  = 280;
constexpr unsigned int CANVAS_HEIGHT = 280;

constexpr unsigned int OUTPUT_WIDTH  = 28;
constexpr unsigned int OUTPUT_HEIGHT = 28;

constexpr unsigned int BUTTON_HEIGHT = 50;
constexpr unsigned int BRUSH_RADIUS  = 8;

constexpr unsigned int CANVAS_OFFSET_Y = BUTTON_HEIGHT;

// ------------------------------------------------------------
// Drawing application
// ------------------------------------------------------------

class DrawingApp
{
public:

    DrawingApp()
        : window(
            sf::VideoMode({WINDOW_WIDTH, WINDOW_HEIGHT}),
            "Grayscale Drawing"
          ),
          texture(
            sf::Vector2u({CANVAS_WIDTH, CANVAS_HEIGHT})
          ),
          sprite(texture)
    {
        window.setFramerateLimit(60);

        clearCanvas();

        // Clear button
        clearButton.setSize({
            150.f,
            static_cast<float>(BUTTON_HEIGHT)
        });

        clearButton.setPosition({20.f, 0.f});
        clearButton.setFillColor(sf::Color(200, 200, 200));

        // Save button
        saveButton.setSize({
            150.f,
            static_cast<float>(BUTTON_HEIGHT)
        });

        saveButton.setPosition({190.f, 0.f});
        saveButton.setFillColor(sf::Color(180, 220, 180));
    }

    void run()
    {
        while (window.isOpen())
        {
            processEvents();
            render();
        }
    }

private:

    sf::RenderWindow window;

    sf::Texture texture;
    sf::Sprite sprite;

    sf::RectangleShape clearButton;
    sf::RectangleShape saveButton;

    // --------------------------------------------------------
    // Pixel buffers
    // --------------------------------------------------------

    // Drawing canvas:
    // 280 x 280 = 78400 pixels
    //
    // 0.0f = black
    // 1.0f = white
    //
    // Row-major:
    // pixels[y * CANVAS_WIDTH + x]

    std::vector<float> pixels;

    // Output:
    // 28 x 28 = 784 pixels
    //
    // Stored as a flat float array.

    std::vector<float> outputPixels;

    bool drawing = false;

    sf::Vector2i previousMousePosition;

    // --------------------------------------------------------
    // Canvas
    // --------------------------------------------------------

    void clearCanvas()
    {
        pixels.assign(
            CANVAS_WIDTH * CANVAS_HEIGHT,
            0.0f
        );

        updateTexture();
    }

    void updateTexture()
    {
        std::vector<std::uint8_t> imageData(
            CANVAS_WIDTH * CANVAS_HEIGHT * 4
        );

        for (unsigned int y = 0; y < CANVAS_HEIGHT; ++y)
        {
            for (unsigned int x = 0; x < CANVAS_WIDTH; ++x)
            {
                const std::size_t index =
                    y * CANVAS_WIDTH + x;

                const std::uint8_t value =
                    static_cast<std::uint8_t>(
                        std::clamp(pixels[index], 0.0f, 1.0f)
                        * 255.0f
                    );

                const std::size_t rgbaIndex = index * 4;

                imageData[rgbaIndex + 0] = value;
                imageData[rgbaIndex + 1] = value;
                imageData[rgbaIndex + 2] = value;
                imageData[rgbaIndex + 3] = 255;
            }
        }

        texture.update(imageData.data());
    }

    // --------------------------------------------------------
    // Drawing
    // --------------------------------------------------------

    void drawPoint(sf::Vector2i position)
    {
        const int canvasX = position.x;

        const int canvasY =
            position.y - static_cast<int>(CANVAS_OFFSET_Y);

        for (int dy = -static_cast<int>(BRUSH_RADIUS);
             dy <= static_cast<int>(BRUSH_RADIUS);
             ++dy)
        {
            for (int dx = -static_cast<int>(BRUSH_RADIUS);
                 dx <= static_cast<int>(BRUSH_RADIUS);
                 ++dx)
            {
                if (dx * dx + dy * dy >
                    static_cast<int>(
                        BRUSH_RADIUS * BRUSH_RADIUS
                    ))
                {
                    continue;
                }

                const int x = canvasX + dx;
                const int y = canvasY + dy;

                if (x < 0 || x >= static_cast<int>(CANVAS_WIDTH))
                    continue;

                if (y < 0 || y >= static_cast<int>(CANVAS_HEIGHT))
                    continue;

                const std::size_t index =
                    static_cast<std::size_t>(y) * CANVAS_WIDTH + x;

                // Draw white.
                pixels[index] = 1.0f;
            }
        }
    }

    void drawLine(sf::Vector2i start, sf::Vector2i end)
    {
        const float dx =
            static_cast<float>(end.x - start.x);

        const float dy =
            static_cast<float>(end.y - start.y);

        const float distance =
            std::sqrt(dx * dx + dy * dy);

        const int steps =
            std::max(1, static_cast<int>(std::ceil(distance)));

        for (int i = 0; i <= steps; ++i)
        {
            const float t =
                static_cast<float>(i) / steps;

            sf::Vector2i position{
                static_cast<int>(std::round(
                    start.x + dx * t
                )),
                static_cast<int>(std::round(
                    start.y + dy * t
                ))
            };

            drawPoint(position);
        }
    }

    // --------------------------------------------------------
    // Downsampling
    // --------------------------------------------------------

    std::vector<float> downsampleTo28x28()
    {
        std::vector<float> output(
            OUTPUT_WIDTH * OUTPUT_HEIGHT,
            1.0f
        );

        constexpr unsigned int SCALE =
            CANVAS_WIDTH / OUTPUT_WIDTH;

        for (unsigned int outY = 0;
             outY < OUTPUT_HEIGHT;
             ++outY)
        {
            for (unsigned int outX = 0;
                 outX < OUTPUT_WIDTH;
                 ++outX)
            {
                float sum = 0.0f;

                for (unsigned int dy = 0;
                     dy < SCALE;
                     ++dy)
                {
                    for (unsigned int dx = 0;
                         dx < SCALE;
                         ++dx)
                    {
                        const unsigned int x =
                            outX * SCALE + dx;

                        const unsigned int y =
                            outY * SCALE + dy;

                        sum += pixels[
                            y * CANVAS_WIDTH + x
                        ];
                    }
                }

                output[
                    outY * OUTPUT_WIDTH + outX
                ] = sum / static_cast<float>(SCALE * SCALE);
            }
        }

        return output;
    }

    // --------------------------------------------------------
    // Saving
    // --------------------------------------------------------

    void savePixels()
    {
        outputPixels = downsampleTo28x28();

        // Save binary float array.
        {
            std::ofstream file(
                "input_image.bin",
                std::ios::binary
            );

            if (!file)
            {
                std::cerr << "Failed to open binary output file.\n";
                return;
            }

            file.write(
                reinterpret_cast<const char*>(
                    outputPixels.data()
                ),
                static_cast<std::streamsize>(
                    outputPixels.size() * sizeof(float)
                )
            );
        }

        // Save CSV for inspection.
        {
            std::ofstream file("input_image.csv");

            if (!file)
            {
                std::cerr << "Failed to open CSV output file.\n";
                return;
            }

            for (unsigned int y = 0;
                 y < OUTPUT_HEIGHT;
                 ++y)
            {
                for (unsigned int x = 0;
                     x < OUTPUT_WIDTH;
                     ++x)
                {
                    file << outputPixels[
                        y * OUTPUT_WIDTH + x
                    ];

                    if (x + 1 < OUTPUT_WIDTH)
                        file << ",";
                }

                file << "\n";
            }
        }

        std::cout
            << "Saved "
            << outputPixels.size()
            << " floats to input_image.bin\n";

        std::cout
            << "Saved 28x28 image to input_image.csv\n";
    }

    // --------------------------------------------------------
    // Input
    // --------------------------------------------------------

    void processEvents()
    {
        while (const std::optional event = window.pollEvent())
        {
            if (event->is<sf::Event::Closed>())
            {
                window.close();
            }

            else if (const auto* mousePressed =
                         event->getIf<sf::Event::MouseButtonPressed>())
            {
                if (mousePressed->button != sf::Mouse::Button::Left)
                    continue;

                sf::Vector2i mousePosition{
                    mousePressed->position.x,
                    mousePressed->position.y
                };

                if (clearButton.getGlobalBounds().contains(
                        sf::Vector2f(mousePosition)))
                {
                    clearCanvas();
                    drawing = false;
                }

                else if (saveButton.getGlobalBounds().contains(
                             sf::Vector2f(mousePosition)))
                {
                    savePixels();
                    drawing = false;
                }

                else if (mousePosition.y >=
                         static_cast<int>(CANVAS_OFFSET_Y) &&
                         mousePosition.x >= 0 &&
                         mousePosition.x < static_cast<int>(CANVAS_WIDTH) &&
                         mousePosition.y <
                         static_cast<int>(CANVAS_OFFSET_Y + CANVAS_HEIGHT))
                {
                    drawing = true;

                    previousMousePosition = mousePosition;

                    drawPoint(mousePosition);
                    updateTexture();
                }
            }

            else if (const auto* mouseReleased =
                         event->getIf<sf::Event::MouseButtonReleased>())
            {
                if (mouseReleased->button ==
                    sf::Mouse::Button::Left)
                {
                    drawing = false;
                }
            }

            else if (const auto* mouseMoved =
                         event->getIf<sf::Event::MouseMoved>())
            {
                if (!drawing)
                    continue;

                sf::Vector2i currentPosition{
                    mouseMoved->position.x,
                    mouseMoved->position.y
                };

                drawLine(
                    previousMousePosition,
                    currentPosition
                );

                previousMousePosition = currentPosition;

                updateTexture();
            }
        }
    }

    // --------------------------------------------------------
    // Rendering
    // --------------------------------------------------------

    void render()
    {
        window.clear(sf::Color(100, 100, 100));

        sprite.setPosition({
            0.f,
            static_cast<float>(CANVAS_OFFSET_Y)
        });

        window.draw(sprite);

        window.draw(clearButton);
        window.draw(saveButton);

        window.display();
    }
};