import { useState, useEffect } from 'react'
import { Search, Loader2, MapPin, ChevronDown } from 'lucide-react'

function App() {
  const [input, setInput] = useState('')
  const [suggestions, setSuggestions] = useState([])
  const [response, setResponse] = useState('')
  const [loading, setLoading] = useState(false)
  const [modelType, setModelType] = useState('transformer') // Default model type

  useEffect(() => {
    const fetchSuggestions = async () => {
      if (input) {
        try {
          const result = await fetch(`http://localhost:5000/api/cities?query=${input}`)
          const data = await result.json()
          setSuggestions(data)
        } catch (error) {
          console.error('Error fetching suggestions:', error)
        }
      } else {
        setSuggestions([])
      }
    }

    const timeoutId = setTimeout(fetchSuggestions, 300)
    return () => clearTimeout(timeoutId)
  }, [input])

  const handleSubmit = async (city) => {
    setLoading(true)
    try {
      const result = await fetch('http://localhost:5000/api/mock-llm', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ 
          city,
          modelType // Include the model type in the API call
        }),
      })
      const data = await result.json()
      setResponse(data.response)
      setInput(city)
      setSuggestions([])
    } catch (error) {
      console.error('Error fetching response:', error)
    }
    setLoading(false)
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 p-6">
      <div className="max-w-3xl mx-auto pt-16">
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold text-gray-900 mb-2">City Explorer</h1>
          <p className="text-gray-600">Discover fascinating insights about cities around the world</p>
        </div>

        <div className="bg-white rounded-2xl shadow-xl p-6 md:p-8">
          <div className="relative">
            <div className="flex space-x-2">
              <div className="relative flex-grow">
                <Search className="absolute left-3 top-3 text-gray-400 w-5 h-5" />
                <input
                  type="text"
                  className="w-full pl-10 pr-4 py-3 text-gray-900 border-2 border-gray-200 rounded-xl 
                            focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-colors
                            placeholder:text-gray-400"
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  placeholder="Enter a city name..."
                />
              </div>
              
              {/* Model type dropdown */}
              <div className="relative min-w-[150px]">
                <select
                  value={modelType}
                  onChange={(e) => setModelType(e.target.value)}
                  className="w-full appearance-none pl-4 pr-10 py-3 text-gray-900 border-2 border-gray-200 rounded-xl
                           focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-colors"
                >
                  <option value="transformer">Transformer</option>
                  <option value="lstm">LSTM</option>
                </select>
                <ChevronDown className="absolute right-3 top-3 text-gray-400 w-5 h-5 pointer-events-none" />
              </div>
            </div>

            {suggestions.length > 0 && (
              <ul className="absolute w-full bg-white mt-2 rounded-xl shadow-lg border border-gray-100 overflow-hidden z-50">
                {suggestions.map((city) => (
                  <li
                    key={city}
                    className="flex items-center px-4 py-3 hover:bg-blue-50 cursor-pointer transition-colors"
                    onClick={() => handleSubmit(city)}
                  >
                    <MapPin className="w-4 h-4 text-gray-400 mr-2" />
                    <span className="text-gray-700">{city}</span>
                  </li>
                ))}
              </ul>
            )}
          </div>

          {loading && (
            <div className="flex items-center justify-center py-8">
              <Loader2 className="w-8 h-8 text-blue-500 animate-spin" />
            </div>
          )}

          {response && (
            <div className="mt-6">
              <div className="bg-gradient-to-r from-blue-50 to-indigo-50 rounded-xl p-6 shadow-sm">
                <div className="flex items-start space-x-4">
                  <div className="flex-shrink-0">
                    <div className="w-10 h-10 rounded-full bg-blue-500 flex items-center justify-center">
                      <MapPin className="w-6 h-6 text-white" />
                    </div>
                  </div>
                  <div>
                    <h3 className="text-lg font-semibold text-gray-900 mb-2">
                      City Insights <span className="text-sm font-normal text-gray-500">({modelType})</span>
                    </h3>
                    <p className="text-gray-700 leading-relaxed">{response}</p>
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

export default App